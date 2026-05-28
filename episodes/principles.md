---
title: "Principles"
teaching: 10 # teaching time in minutes
exercises: 2 # exercise time in minutes
---

:::::::::::::::::::::::::::::::::::::: questions 

- What are key concepts I should know if I want to develop or train a Machine Learning / Artificial Intelligence model?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Explain basic concepts

::::::::::::::::::::::::::::::::::::::::::::::::

## Concept 1: What are Machine Learning and Artificial Intelligence?

We are all familiar with the idea of applying rules to data to gain insights and make decisions. For example, we learn that human body temperature is ~37 °C (~98.5 °F), and that higher or lower temperatures can be cause for concern.

As programmers we understand how to codify these rules. If we were developing software for a hospital to flag patients at risk of deterioration, we might create early-warning rules such as those below:

```python
def has_fever(temp_c):
    if temp_c > 38:
        return True
    else:
        return False
```

With Machine Learning (ML), we modify this approach. Instead, we give data and insights to a framework (or “model”) that can learn the rules and patterns for itself. As the volume and complexity of data increases, so does the value of having models that can generate new rules.

In this sense, having a machine learning model that captures accurately the patterns in the data is the driving force behind the broader field of Artificial Intelligence (AI). To put it simply, if machine learning is the specific tool used to discover rules from data, AI is the overarching umbrella: the grand scientific goal of creating systems that can reason, perceive, and make autonomous decisions like a human expert.

A couple more terms come up when talking about AI: Generative AI (or GenAI) and Artificial General Intelligence (AGI). While traditional machine learning focuses on a narrow set of outcomes or predictions, GenAI creates new content with seemingly free-format: text, images, audio, or video. On the broader horizon lies AGI, the still-theoretical destination of AI research to build systems with broad, human-level intelligence capable of teaching itself new skills and transferring knowledge across completely unrelated scientific domains.

To synthetize, here are some take-home definitions you could use:

- **Machine Learning (ML)**: Techniques where a computer can “learn” patterns in data, usually by being shown numerous examples to train it. 
- **Artificial Intelligence (AI)**: Sometimes used as a synonym of Machine Learning. Currently, used frequently in the context of Generative AI (GenAI), as machine learning models that can generate text, images, or other forms of data. Artificial General Intelligence (AGI) is a type of AI that matches or surpasses human capabilities across virtually all cognitive tasks.

::::::::::::::::::::::::::::::::::::: challenge

Take a couple of minutes to discuss with the person seated next to you or with the entire class: What do you already know about machine learning? This can be anything! News you’ve read, tools you use, concerns you have.

:::::::::::::::::::::::::::::::::::::::::::::::

## Concept 2: Paradigms/Types of Machine Learning

Now that we understand what machine learning is at a high level, how do these models actually go about "learning" those rules? Depending on the nature of your data and the question you are trying to answer, machine learning generally falls into three main paradigms: Supervised Learning, Unsupervised Learning, and Reinforcement Learning.

### Supervised learning

In supervised learning, a model is trained on data that already contains the correct answers, which we call **labels**. By providing the model with both the inputs and the expected outputs, it learns the underlying mathematical relationship that connects them.

Ultimately, the goal is to train a model that accurately maps inputs to outputs, so that it can reliably predict the correct labels when faced with entirely new, unseen data.

For instance, consider your email spam filter: it learns from millions of messages labeled "Spam" or "Not Spam" to automatically flag a brand-new, unseen message. Similarly, to predict a specific number like a home's market value, you can feed a model historical real estate data where the inputs are features like square footage and location, and the labels are the actual sale prices. Once the algorithm learns how those features map to those numbers, it can evaluate a newly listed house and give a prediction on its market value.

:::::::::::::::::::::::::::::::::::::::::: spoiler

#### Large Language Models (LLMs) use supervised learning

One form of AI you maybe have heard about are Large Language Models (LLMs). At their core, they operate on a surprisingly simple mechanism: next-word prediction. When you type a prompt into an LLM, the model does not "think" or understand concepts the way humans do; instead, it uses an ML model that learned patterns from massive datasets to guess the single most likely word to follow your text. By repeating this cycle thousands of times per second, it strings together words one by one to generate fluid, human-like sentences.

::::::::::::::::::::::::::::::::::::::::::::::::::

### Unsupervised learning

In unsupervised learning, the data has no labels, and a model is left on its own to explore the data, find hidden structures, and group similar data points together based purely on their characteristics.

The goal here is to discover underlying patterns, anomalies, or natural groupings within the datasets without any human guidance or predefined answers.

For instance, consider customer segmentation in marketing: a model analyzes purchasing history and browsing habits to automatically group shoppers into distinct personas, like "bargain hunters" or "tech enthusiasts," so companies can target them differently. Notice the personas are not predefined, but can be infered after analyzing how the model grouped them. This is a type of ML model called **clustering**.

Similarly, take a researcher dealing with massive datasets, like genomic sequencing with thousands of genetic markers. A **dimensionality reduction** model is useful here to strip away redundant or uninformative variables while preserving the core variation, making it possible to visualize and analyze high-dimensional data on a simple two-dimensional plot.


### Reinforcement learning

Reinforcement learning is a bit different. It doesn't rely on a static dataset. Instead, an autonomous "agent" learns to interact with a dynamic environment by taking actions and receiving feedback in the form of rewards or penalties, optimizing its behavior over time through trial and error.

Think of this like training a dog: you don't show the dog a spreadsheet of "how to sit," you reward it when it does the right thing and withhold treats when it doesn't. Over time, the dog optimizes its behavior to get the most treats. This is the type of models behind autonomous driving vehicles.

::::::::::::::::::::::::::::::::::::: challenge

Match the Research Task
Look at the three scientific scenarios below. Discuss with a partner which machine learning paradigm (Supervised, Unsupervised, or Reinforcement Learning) would be the right tool for the job:

Scenario A: A climatologist has thousands of satellite images of sea ice cover, some marked by experts as "critical melt" and others as "stable." They want to build a tool that automatically flags new images showing critical melt.

Scenario B: A geneticist has RNA sequencing data from 10,000 cells. They don't know how many cell types are in the sample, but they want to see if the cells naturally group into distinct, unknown sub-populations.

Scenario C: A drone engineer wants to program an autonomous quadcopter to navigate through a dense, unpredictable forest canopy without crashing, adjusting its trajectory mid-flight based on wind gusts.

:::::::::::::::::::: solution

Scenario A is Supervised Learning. The data is labeled (marked as "critical" vs "stable"), and the goal is to predict those specific categories for new data.

Scenario B is Unsupervised Learning. The data is unlabeled (the cell types are unknown), and the goal is to find hidden clusters or groupings.

Scenario C is Reinforcement Learning. The drone is an agent interacting with a dynamic environment (the forest), learning through trial and error to maximize a reward (reaching the destination without crashing).

::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::::::::::

## Concept 3: Clearly identify your goal and available data

For the remaining of this lesson, we will focus on a particular example: the Palmer penguins dataset.



- Start coding with penguins dataset: Load data and explore

## Concept 5: Supervised learning tasks

- Prediction and regression

## Concept 4: ML models can only use numbers 

- Code: Drop non-numeric data


## Concept 6: How ML models learn

- Talk about objective function, loss, learning
- It learns patterns from existing data, with its bias

## Concept 7: Splitting the data


## Concept 8: Evaluating your model


## Concept 9: Experiment and iterate

- AI/ML is a computer algorithm learning trends from previous/past data/observations.
- Three types: supervised, unsupervised, reinforcement learning.
- Focus on supervised learning as it is the most widely used and most useful for all types of fields
- Supervised takes input data/features to predict with this information an output variable/target/label



- Identify your problem, task, inputs, and outputs, what you want to predict
- Clean your data and perform necessary data preparation
- Split your data into train and test
- Choose an algorithm to train from scratch or a pretrained model
- Choose performance metrics to know how to evaluate your model
- Experiment and iterate

## Most important concepts:
- Task: Regression, classification
- Models only take numbers, clean and prepare your data
- Model learns from seen data, with its bias and its historic trends
- Loss function, guessing at first, learning, and reducing loss
- Metrics and performance
- Garbage in, garbage out
- Train/test split
- Overfitting




::::::::::::::::::::::::::::::::::::: keypoints 

- Review the principles

::::::::::::::::::::::::::::::::::::::::::::::::