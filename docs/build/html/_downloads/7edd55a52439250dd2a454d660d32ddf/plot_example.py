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
# In `Callaghan and Müller-Hansen, 2020 <https://doi.org/10.1186/s13643-020-01521-4>`_
# we offered a theoretically well motivated stopping
# criteria, which we demonstrated was safe to use. It allows you to communicate
# your confidence in achieving any arbitrary recall target. This package aims to
# make this stopping criteria easy to use for python users.

# %%
# Data
# ------------------------
#
# Lets initialise some test data. The `generate_data` function takes a number
# of documents, as well as the  the prevalence of relevant documents, or uses
# default values, and simulates a prioritised screening-like
# process, where we sample documents, where we are `bias` times more likely to select a random
# relevant document than a random irrelevant document.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from buscarpy import generate_dataset

df = generate_dataset(random_seed=12345)
df.relevant.cumsum().plot()
df.head()

# %%
# When is it safe to stop?
# ------------------------
#
# Let's imagine we've seen just the first 10,000 documents. We can use our stopping criteria
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
# We can calculate this using the buscarR package, by passing a list of 1s and 0s
# to our `calculate_h0` function, along with the total number of documents in
# in the dataset. The list of 1s and 0s represents INCLUDE and EXCLUDE decisions
# by human screeners, and should be in the order in which the documents were
# screened


from buscarpy import calculate_h0
seen_documents = df['relevant'][:10000]

fig, ax = plt.subplots()
seen_documents.cumsum().plot()
ax.set_xlim(xmax=df.shape[0])

calculate_h0(seen_documents, df.shape[0])

# %%
# Our p score indicates that we are not yet confident enough to stop screening.
# If we "see" an additional 2,000 documents, this will change

seen_documents = df['relevant'][:12000]

fig, ax = plt.subplots()
seen_documents.cumsum().plot()
ax.set_xlim(xmax=df.shape[0])

calculate_h0(seen_documents, df.shape[0])

# %%
# We can now be **very confident** that we have **not missed** our recall target

# %%
# Changing recall targets
# ------------------------
#
# We can calculate the same stopping criteria for a different **recall target**,
# simply by using the `recall_target` argument in `calculate_h0`.

calculate_h0(seen_documents, df.shape[0], recall_target=0.99)

# %%
# If we increase the recall target, we become **less** confident that we
# have **not missed** our target.
#
# In many practical cases, we may not be very confident in one target, but much
# more confident in a target that is only a little smaller. The `recall_frontier`
# function calculates and plots the p score for several different recall targets,
# helping to inform and transparently communicate our decision about the safety
# of stopping screening at any given point.

from buscarpy import recall_frontier

recall_frontier(seen_documents, df.shape[0])

# %%
# Retrospective stopping criteria
# ------------------------
#
# The package also includes a helper function to calculate the stopping criteria
# at each point on a curve that has already been seen. By default we calculate
# this after each batch of 1,000 documents. Change the `batch_size` to alter this,
# though be warned that reducing it will increase the number of calculations that
# needs to be made.

from buscarpy import retrospective_h0

retrospective_h0(seen_documents, df.shape[0])

# %%
# Biased urns
# ------------------------
#
# The stopping criteria, which is described in
# `Callaghan and Müller-Hansen, 2020 <https://doi.org/10.1186/s13643-020-01521-4>`_
# assumes that the documents we have screened previously were drawn *at random*
# from the remaining records. This assumption is *conservative*, as the
# machine-learning process should make it more likely that we pick a relevant
# document than an irrelevant document.
#
# Being conservative, it is safe to use this stopping criteria (and evaluations
# show that it is wrong less than 5% of the time if the confidence level is set
# to 95%), but its conservative nature means that we will stop later than we
# strictly need to.
#
# Biased urn theory offers us a more realistic set of assumptions, as it
# describes the probability distribution given a situation where we are more
# likely to select one type of item than another. We can implement this in
# buscarpy, by setting the `bias` parameter of our functions. `bias` describes
# how much more likely it is to select a relevant than a non-relevant document.
#
# However, estimating this parameter is non-trivial, and work
# on how to do this safely is currently ongoing.

h0_1 = retrospective_h0(seen_documents, df.shape[0], bias=1, plot=False)
h0_5 = retrospective_h0(seen_documents, df.shape[0], bias=5, plot=False)

fig, ax = plt.subplots()

ax.plot(seen_documents.cumsum())
ax2 = ax.twinx()
ax2.scatter(h0_1['batch_sizes'], h0_1['p'], label='Unbiased')
ax2.scatter(h0_5['batch_sizes'], h0_5['p'], label='Bias==5')
ax.set_xlabel('Documents screened')
ax.set_ylabel('Relevant documents identified')
ax2.set_ylabel('p score for H0 that recall target missed')

ax2.legend()
