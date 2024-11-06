## Ömer Özer NLP Repo

#### This repo contains my attempts to create encoder and decoder architectures using PyTorch. While writing the codes, I got help from Andrej Karpathy's videos and the codes he wrote.

### Datasets

For simpler models and first attempts, I used the _input.txt_ file, which contains a small excerpt from Shakespeare. I generally use this txt file as a dataset when I want to deliberately overfit the models and code to see if there are any errors and to measure their learning capacity.

For complex models and long learning processes, I downloaded the 10B version of the edu fineweb dataset from hugging face and processed it for various purposes. The _fineweb.py_ file can be used to download the dataset, tokenize it using the gpt2 tokenizer, and split it into chunks containing 100 million tokens each. The _process_fineweb_for_encoder.py_ file was created to prepare the dataset required for encoder training. It applies random masking to tokens.

### Training

#### There are 3 different Python codes for training:

-   The _train_decoder.py_ file contains decoder only architecture and training of the model. This code is finalized and ready to be used. I ran a few training experiments using GPUs in Google Colab. Although the statistics I received were promising, I could not create a model at the level I wanted because the computing power was not enough.

-   The _train_encoder_decoder.py_ and _train_encoder.py_ files were designed to contain the encoder-decoder and encoder only structure, respectively. I'm trying to create it by making changes via _train_decoder.py_, but the codes are not finished and ready to run yet. Since it is more difficult to prepare the dataset of the encoder structure and make it ready to be given as input to the model, I have not fully finished it yet.

#### Inference

I created the generate.py file to test the decoder only models I created using the train_decoder.py file and observe their outputs. As a starting point, it takes the sentence "Hello, As a Language Model" and tokenizes it and gives it to the model as the beginning of the sequence. It prints the output of the model at the end.
