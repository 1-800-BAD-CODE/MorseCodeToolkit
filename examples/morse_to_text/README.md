
# Morse-to-text
This example shows how to

* Prepare data for model training and evaluation
* Train a CTC-based NeMo morse recognition model with synthetic training data
* Evaluate a morse recognition model against a synthetic test set

The remainder of this document assumes we have some corpora directory
where we store all our data, and a path to the git base of this 
project:

```console
$ corpus_dir=/path/to/corpora/
$ mct_root=/path/to/morsecodetoolkit/
```

## Language
This README describes training an English model. To train Russian
instead of English, 

1. Replace the English text data with Russian
2. Use the configuration `quartznet_5x3_russian.yaml`

One caveat with Russian is that there is no `Ё` defined in Russian
morse code. In the text data, all instances of `Ё` should be replaced
with `Е` (analogous for lowercase). See the Wikipedia entry for more details: 
[Russian morse](https://en.wikipedia.org/wiki/Russian_Morse_code#Table_&_Melody).

To train in another language, see the documentation on 
[adding an alphabet](../../docs/alphabets_and_encodings.md) 
and create a new configuration based on one of the examples here.

# Data Prep
This section describes how to prepare input data for this recipe. We
need four files:

* Text for synthesizing training data
* Audio for training background noise
* Similar text and audio for a dev set

## LibriSpeech Prep
We will use the following portions of LibriSpeech for the specified
purposes:

* `dev-other` text and audio: Dev set text and background noise 
* `dev-clean` audio: Training background noise
* `test-clean` text and audio: Evaluating the final model

For information on this corpus, see the 
[openslr page](https://openslr.org/12/).

### Download and Extract Libri
If LibriSpeech is not already locally downloaded, we do so now:

```console
$ cd ${corpus_dir}
$ for name in "dev-other" "dev-clean" "test-clean" ; do
      wget https://www.openslr.org/resources/12/${name}.tar.gz
      tar xf ${name}.tar.gz  
  done
```

We should now have a corpus structure similar to

```console 
$ tree -L 1 ${corpus_dir}/LibriSpeech/
    <corpus_dir>/LibriSpeech/
    ├── dev-clean
    ├── dev-other
    └── test-clean
```

### Generate Libri Audio Manifests
Next need to generate manifests in a format interpretable by the data
set class for use as background noise. We can do that for each partition
with an included script:

```console
$ for name in "dev-other" "dev-clean" "test-clean" ; do
      python ${mct_root}/scripts/make_audio_manifest.py \
          ${corpus_dir}/LibriSpeech/name/ \
          --output-file ${corpus_dir}/LibriSpeech/${name}.json \
          --file-ext flac 
  done
```

### Generate Libri Text
We compile all the text data for each partition into simple text files
with one sentence per line and no metadata. For LibriSpeech, this means
just concatenating all `*.txt` files and stripping the first field, which
is the speaker ID:

```console
$ cd ${corpus_dir}/LibriSpeech/
$ for name in "dev-other" "dev-clean" "test-clean" ; do
      find ./${name} -name '*.txt' \
        -exec cut -d ' ' -f 2- {} \; > ${name}.txt
  done
  
$ head -2 dev-other.json 
  {"audio_filepath": "<corpus_dir>/LibriSpeech/dev-other/3660/172183/3660-172183-0001.flac"}
  {"audio_filepath": "<corpus_dir>/LibriSpeech/dev-other/3660/172183/3660-172183-0011.flac"}
```

Now we should have three .txt files in the LibriSpeech directory:

```console
$ tree -L 1 ${corpus_dir}/LibriSpeech/
    <corpus_dir>/LibriSpeech/
    ├── dev-clean
    ├── dev-clean.json
    ├── dev-clean.txt
    ├── dev-other
    ├── dev-other.json
    ├── dev-other.txt
    ├── test-clean
    ├── text-clean.json
    └── test-clean.txt
    
$ head -2 ${corpus_dir}/LibriSpeech/dev-other.txt 
  GERAINT AS HE HAD BEEN USED TO DO WHEN HE WAS AT ARTHUR'S COURT FREQUENTED TOURNAMENTS
  BEFORE GERAINT THE SCOURGE OF THE ENEMY I SAW STEEDS WHITE WITH FOAM AND AFTER THE SHOUT OF BATTLE A FEARFUL TORRENT
```
 
## Prepare News-Commentary Text
Prepare the training text by downloading and extracting some text data from
WMT, and set an environment variable we can reference later. For 
information about this data, including license, see the 
[WMT readme](http://data.statmt.org/news-commentary/README).

```console
$ cd ${corpus_dir}
$ wget http://data.statmt.org/news-commentary/v16/training-monolingual/news-commentary-v16.en.gz
$ gunzip news-commentary-v16.en.gz
```



# Model Training
In this section we will train a morse recognition model using synthetic
data. We will use the NeMo toolkit to train a small QuartzNet that
targets the English alphabet. The texts used to generate morse signals
comes from the WMT and LibriSpeech corpora, while the background noise
comes from LibriSpeech audio.

```console
python train.py \
    --config-path $PWD/conf/ \
    --config-name quartznet_10x5 \
    train_text=${corpus_dir}/news-commentary-v16.en 
    train_background_manifest=${corpus_dir}/LibriSpeech/dev-clean/manifest.json 
    dev_text=${corpus_dir}/LibriSpeech/dev-clean/all.txt \
    dev_background_manifest=${corpus_dir}/LibriSpeech/dev-clean/manifest.json
```

This should reproduce the model that can be found in `pretrained/english/quartznet10x5.nemo`
and should take a few hours on a single GPU to get a WER
of less than 1%.

To fine-tune a pre-trained model, add to this argument
the flag `++init_from_nemo_model=../../pretrained/english/quartznet10x5.nemo`.
 

# Model Evaluation
The script `evaluate.py` can evaluate a trained model against a 
corpus of morse signals, which can be synthesized with this project.

## Preparing Data for Evaluation
For details on synthesizing a dataset, see the `synthesize_dataset`
example. The following steps briefly describe the process.

First, prepare some data to be use in generation.
A preprocessed LibriSpeech dev is used here:

```text
$ libri_text=/path/to/librispeech/dev-clean/all.txt
$ libri_audio_manifest=/path/to/librispeech/dev-clean/manifest.json

$ head -2 ${libri_text}
ARDENT IN THE PROSECUTION OF HERESY CYRIL AUSPICIOUSLY OPENED HIS REIGN BY OPPRESSING THE NOVATIANS THE MOST INNOCENT AND HARMLESS OF THE SECTARIES
WITHOUT ANY LEGAL SENTENCE WITHOUT ANY ROYAL MANDATE THE PATRIARCH AT THE DAWN OF DAY LED A SEDITIOUS MULTITUDE TO THE ATTACK OF THE SYNAGOGUES

$ head -2 ${libri_audio_manifest}
{"audio_filepath": "/path/to/librispeech/dev-clean/wav/5338-284437-0020.wav"}
{"audio_filepath": "/path/to/librispeech/dev-clean/wav/6345-93302-0008.wav"}
```

Here, `libri_text` is a plain text file containing all the LibriSpeech dev
text, one sentence per line. It will be used to generate morse.

`libri_audio_manifest` is a list of JSON entries, one per line, each 
describing one audio file. These audio files will be drawn randomly and
used as background noise to prevent clean morse signals.

Using these variable, we can use the scripts in the example
`synthesize_dataset`:

```text
$ cd examples/synthesize_dataset
$ python synthesize.py \
    text_file=${libri_text} \
    audio_manifest=${libri_audio_manifest} \
    output_dir=/pathto/synthetic/dataset/
```

## Running Evaluation
With a pre-trained model and a prepared manifest containing synthesized
morse signals, we run the `evaluate.py` script:

```text
$ model=../../pretrained/english/quartznet10x5.nemo
$ manifest=/path/to/synthetic/dataset/manifest.json
$ python evaluate.py ${model} ${manifest}
    ...
    INFO : WER: 2.8%
```

The WER is 2.8% here, although this example is not meant to be authoritative
on model evaluation or morse corpus generation. Rather the intent here is to
demonstrate the general use.