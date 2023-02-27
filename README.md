# Train a sarcasm detector with retraining loop using Transformers, ClearML and Gradio

Watch the video here: https://youtu.be/0wmwnNnN8ow

# Setup

## Python

This was run using Python 3.10.9, simply create a new virtual environment and run `pip install -r requirements.txt`. Please let me know if this does not work for you so I can update the requirements file :)

## Data

The data is located in the `data` folder, it is a slightly preprocessed version of [this](https://www.kaggle.com/datasets/danofer/sarcasm) dataset. Below is the citation of the authors for maximum exposure for them, they did an awesome job and thanks to them for releasing it!

```
@unpublished{SARC,
  authors={Mikhail Khodak and Nikunj Saunshi and Kiran Vodrahalli},
  title={A Large Self-Annotated Corpus for Sarcasm},
  url={https://arxiv.org/abs/1704.05579},
  year=2017
}
```

Next, in order to work properly, this data has to be read into a ClearML Data Dataset. You only need 3 commands:

```
$ clearml-data create --project sarcasm_detector --name sarcasm_dataset
...
$ clearml-data add --files data
...
$ clearml-data close
...
```

## Training

You can train a logistic regression model as well as a transformer model using respectively `train_sklearn.py` and `train_transformer.py`. These models are not optimized, not a lot of time went into them and they are very basic.
The models will be trained on a subset of the data (N=1000), but you can edit that out in each file if you want to train on the whole dataset.

## Gradio

There are 3 gradio apps in increasing stages of complexity.
`gradio_app.py`: very basic, runs both models on an input sentence, path to models hardcoded, so you'll have to change this.
`gradio_advanced.py`: almost the same app, only now using the gradio blocks api (was a learning experience for me, but left it in here)
`gradio_product.py`: the "polished" app for product people, inludes the labeling tool as a reference implementation, is meant more for illustrative purposes on how to use the Clearml Data api.

In the latter 2 apps, you'll need to provide the Clearml model ID for the app to load. You can find this ID in the webUI under the experiment that trained the model in the artifacts tab.

## Closing the loop

Now that you have extra data in a new version of the dataset using gradio, you can close the loop by rerunning the training from within ClearML using the clearml agent.

Run the following on any machine:
```
# set up python env first!
pip install clearml clearml-agent
clearml-init
clearml-agent daemon --queue "default" --foreground --docker python:3.10
```

Now you should have a worker that can execute your experiments. Go to your training experiment, right click, clone and then right click enqueue to put it in a queue (in this case "default").