"""
Calculate stopping criteria
===========================

"""

# %%
# Introduction
# ------------------------
#
# When we do machine learning-prioritised screening in a systematic review, we
# can only save work if we have a method for deciding when to stop.
# Several methods have been suggested, but those which rely on rules of thumb
# (like stopping after N consecutive irrelevant records) are not supported by either
# theory or empirical evaluations. A particular value of N may work well in one
# review, but do badly in another. The right value depends on the size of the dataset,
# the prevalence of relevant documents, the effectiveness of the machine learning
# algorithm, and a bit of luck. Moreover, using such a criterion does not allow
# us to say anything about our expected recall, nor our confidence in achieving it.
# In callaghan_statistical_2020 we offered a theoretically well motivated stopping
# criteria, which we demonstrated was safe to use. It allows you to communicate
# your confidence in achieving any arbitrary recall target. This package aims to
# make this stopping criteria easy to use for R users.

# %%
# Data
# ------------------------
#
# Lets initialise some test data. For demonstration purposes, we define the number of documents,
# and the prevalence of relevant documents. Then we simulate a prioritised screening-like
# process, where we sample documents, where we are `bias` times more likely to select a random
# relevant document than a random irrelevant document.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n_docs = 6000
n_relevant_docs = 1000
urn_bias = 20

sample = np.zeros(n_docs)
sample[:n_relevant_docs] = 1

ps = np.ones(n_docs)
ps[:n_relevant_docs] = urn_bias
ps = ps/ps.sum()

sample = np.random.choice(sample,sample.shape,replace=False, p=ps)
df = pd.DataFrame({'relevance': sample})
df['id'] = df.index
df.relevance.cumsum().plot()
df.head()

# %%
# When is it safe to stop?
# ------------------------
#
# Let's imagine we've seen just the first 20,000 documents. We can use our stopping criteria
# to calculate a p score for a null hypothesis that we have missed so many documents that we have
# not achieved our recall target.
#
# If the p score is low, then we can reject that null hypothesis and stop safely.
# The lower the score, the more confident we can be about doing this. The p score
# is given by calculating the probability of observing the previous sequence
# of relevant and irrelevant documents, if there were enough remaining relevant
# documents to mean that our recall target had not been achieved.
#
# For example, if we have seen 95 relevant documents,
# and our recall target is 95%, then there would have to be at least 6 relevant documents
# remaining for us to have missed our target. If we have just observed a sequence of 100 irrelevant
# document in a row, we ask how likely it would be to observe that by random sampling,
# if there were 6 relevant documents remaining.
#
# We can calculate this using the buscarR package, by passing dataframe with a column
# `relevant` that contains 1s and 0s for relevant and irrelevant documents (and NAs for
# documents we have not seen yet). A separate column `seen` tells us if this document
# has been seen by a human yet or not. The dataframe should have as many rows as there
# are unique documents in the dataset, should contain all that have been seen by a human
# and all documents that have not yet been seen by a human. The human-screened documents
# should be in the order in which they were screened.


from buscarpy import calculate_h0
documents_seen = 1500
df['seen_relevance'] = np.NaN
df.loc[:documents_seen,'seen_relevance'] = df.loc[:documents_seen,'relevance']
fig, ax = plt.subplots()
df.seen_relevance.cumsum().plot()
ax.set_xlim(xmax=df.shape[0])

calculate_h0(df)
