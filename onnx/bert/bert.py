"""
This code is for:
https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/notebooks/PyTorch_Bert-Squad_OnnxRuntime_CPU.ipynb
"""
# bert.py

import os

cache_dir = os.path.join(".", "cache_models")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# SQuAD = Stanford Question Answering Dataset: NLP understanding benchmark
predict_file_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
predict_file = os.path.join(cache_dir, "dev-v1.1.json")
if not os.path.exists(predict_file):
    import wget
    print("Start downloading predict file.")
    wget.download(predict_file_url, predict_file)
    print("Predict file downloaded.")

# For fine tuned large model, the model name is "bert-large-uncased-whole-word-masking-finetuned-squad". Here we use bert-base for demo.
model_name_or_path = "bert-base-cased"
max_seq_length = 128
doc_stride = 128
max_query_length = 64

# Enable overwrite to export onnx model and download latest script each time when running this notebook.
# enable_overwrite = True
enable_overwrite = False

# Total samples to inference. It shall be large enough to get stable latency measurement.
total_samples = 100
#### ------------------------------- ####

def main():
    # The following code is adapted from HuggingFace transformers
    # https://github.com/huggingface/transformers/blob/master/examples/run_squad.py

    from transformers import (BertConfig, BertForQuestionAnswering, BertTokenizer)

    # Load pretrained model and tokenizer
    config = BertConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path, do_lower_case=True, cache_dir=cache_dir)
    model = BertForQuestionAnswering.from_pretrained(model_name_or_path,
                                        from_tf=False,
                                        config=config,
                                        cache_dir=cache_dir)
    
    # load some examples
    from transformers.data.processors.squad import SquadV1Processor

    # Preprocessor for the SQuAD v1.1 dataset
    processor = SquadV1Processor()
    examples = processor.get_dev_examples(None, filename=predict_file)

    from transformers import squad_convert_examples_to_features

    features, dataset = squad_convert_examples_to_features( 
                examples=examples[:total_samples], # convert just enough examples for this notebook
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                doc_stride=doc_stride,
                max_query_length=max_query_length,
                is_training=False,
                return_dataset='pt'
            )

    output_dir = os.path.join(".", "onnx_models")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)   
    export_model_path = os.path.join(output_dir, 'bert-base-cased-squad.onnx')

    import torch
    device = torch.device("cpu")

    # Get the first example data to run the model and export it to ONNX
    data = dataset[0]
    inputs = {
        'input_ids':      data[0].to(device).reshape(1, max_seq_length),
        'attention_mask': data[1].to(device).reshape(1, max_seq_length),
        'token_type_ids': data[2].to(device).reshape(1, max_seq_length)
    }

    # Set model to inference mode, which is required before exporting the model because some operators behave differently in 
    # inference and training mode.
    model.eval()
    model.to(device)

    if enable_overwrite or not os.path.exists(export_model_path):
        with torch.no_grad():
            symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
            torch.onnx.export(model,                                          # model being run
                            args=tuple(inputs.values()),                      # model input (or a tuple for multiple inputs)
                            f=export_model_path,                              # where to save the model (can be a file or file-like object)
                            opset_version=14,                                 # the ONNX version to export the model to
                            do_constant_folding=True,                         # whether to execute constant folding for optimization
                            input_names=['input_ids',                         # the model's input names
                                        'input_mask', 
                                        'segment_ids'],
                            output_names=['start', 'end'],                    # the model's output names
                            dynamic_axes={'input_ids': symbolic_names,        # variable length axes
                                          'input_mask' : symbolic_names,
                                          'segment_ids' : symbolic_names,
                                          'start' : symbolic_names,
                                          'end' : symbolic_names})
            print("Model exported at ", export_model_path)
    
    import time

    # Measure the latency. It is not accurate using Jupyter Notebook, it is recommended to use standalone python script.
    latency = []
    with torch.no_grad():
        for i in range(total_samples):
            data = dataset[i]
            inputs = {
                'input_ids':      data[0].to(device).reshape(1, max_seq_length),
                'attention_mask': data[1].to(device).reshape(1, max_seq_length),
                'token_type_ids': data[2].to(device).reshape(1, max_seq_length)
            }
            start = time.time()
            outputs = model(**inputs)
            latency.append(time.time() - start)
    print("PyTorch {} Inference time = {} ms".format(device.type, format(sum(latency) * 1000 / len(latency), '.2f')))

    import onnxruntime
    import numpy

    sess_options = onnxruntime.SessionOptions()

    # Optional: store the optimized graph and view it using Netron to verify that model is fully optimized.
    # Note that this will increase session creation time, so it is for debugging only.
    sess_options.optimized_model_filepath = os.path.join(output_dir, "optimized_model_cpu.onnx")

    # Specify providers when you use onnxruntime-gpu for CPU inference.
    session = onnxruntime.InferenceSession(export_model_path, sess_options, providers=['CPUExecutionProvider'])

    latency = []
    for i in range(total_samples):
        data = dataset[i]
        ort_inputs = {
            'input_ids':  data[0].cpu().reshape(1, max_seq_length).numpy(),
            'input_mask': data[1].cpu().reshape(1, max_seq_length).numpy(),
            'segment_ids': data[2].cpu().reshape(1, max_seq_length).numpy()
        }
        start = time.time()
        ort_outputs = session.run(
            None, 
            ort_inputs
        )
        latency.append(time.time() - start)
    print("OnnxRuntime cpu Inference time = {} ms".format(format(sum(latency) * 1000 / len(latency), '.2f')))

if __name__ == "__main__":
    main()