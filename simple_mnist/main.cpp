#include <fstream>
#include <map>
#include <chrono>
#include "logging.h"
#include "NvInfer.h"

using namespace nvinfer1;
#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)
// stuff we know about the network and the input/output blobs
static const int INPUT_W = 28;
static const int INPUT_H = 28;
static const int INPUT_SIZE = 784;
static const int OUTPUT_SIZE = 10;

const char *INPUT_BLOB_NAME = "data";
const char *OUTPUT_BLOB_NAME = "prob";

static Logger gLogger;
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t *val = reinterpret_cast<uint32_t *>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

void freeWeightsMap(std::map<std::string, Weights> &weightMap)
{
    for (auto it : weightMap)
    {
        free((void *)it.second.values);
    }
}

ICudaEngine *createEngine(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt)
{
    INetworkDefinition *network = builder->createNetworkV2(0U);

    // Create input tensor of shape { 1, 1, 32, 32 } with name INPUT_BLOB_NAME
    ITensor *data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{1, INPUT_H, INPUT_W});
    assert(data);
    std::map<std::string, Weights> weightMap = loadWeights("../mnist.weights");
    IFullyConnectedLayer *fc1 = network->addFullyConnected(*data, 256, weightMap["model.fc1.weight"], weightMap["model.fc1.bias"]);
    assert(fc1);
    IActivationLayer *relu1 = network->addActivation(*fc1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IFullyConnectedLayer *fc2 = network->addFullyConnected(*relu1->getOutput(0), 256, weightMap["model.fc2.weight"], weightMap["model.fc2.bias"]);
    assert(fc2);
    IActivationLayer *relu2 = network->addActivation(*fc2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    IFullyConnectedLayer *fc3 = network->addFullyConnected(*relu2->getOutput(0), 10, weightMap["model.out.weight"], weightMap["model.out.bias"]);
    assert(fc3);

    // ISoftMaxLayer* softmax = network->addSoftMax(*fc3->getOutput(0));
    // assert(softmax);
    fc3->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out" << std::endl;
    network->markOutput(*fc3->getOutput(0));

    // build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1 << 32);
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

    // don't need the network anymore
    network->destroy();

    // Release host memory
    for (auto &mem : weightMap)
    {
        free((void *)(mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory **modelStream)
{
    // Create builder
    IBuilder *builder = createInferBuilder(gLogger);
    IBuilderConfig *config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine *engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext &context, float *input, float *output, int batchSize)
{
    const ICudaEngine &engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void *buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char const *argv[])
{
    if (argc != 2)
    {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./alexnet -s   // serialize model to plan file" << std::endl;
        std::cerr << "./alexnet -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s")
    {
        IHostMemory *modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("alexnet.engine", std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    }
    else if (std::string(argv[1]) == "-d")
    {
        std::ifstream file("alexnet.engine", std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    }
    else
    {
        return -1;
    }

    // Subtract mean from image
    float data[INPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE; i++)
        data[i] = 1;

    IRuntime *runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);

    // Run inference
    float prob[OUTPUT_SIZE];
    for (int i = 0; i < 100; i++)
    {
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, 1);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    std::cout << "\nOutput:\n\n";
    for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    {
        std::cout << prob[i] << ", ";
    }
    std::cout << std::endl;
    return 0;
}
