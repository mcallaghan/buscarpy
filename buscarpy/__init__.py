import math
from scipy.stats import hypergeom
import pandas as pd
import numpy as np

def k_h0(r_al, r_seen, recall_target):
    """
    Calculate the smallest number of relevant documents there would have to be in our urn, for us to have missed a given recall recall_target

    :param r_seen: the number of relevant documents seen.
    :type kind: int
    :return: The number smallest number of relevant documents that could be in our urn that there could be if we had missed our recall target
    :rtype: int

    """
    return math.floor(r_seen/recall_target-r_al+1)


def calculate_h0(df, recall_target=.95):
    """
    Calculate the smallest number of relevant documents there would have to be in our urn, for us to have missed a given recall recall_target

    :param r_seen: the number of relevant documents seen.
    :type kind: int
    :return: The number smallest number of relevant documents that could be in our urn that there could be if we had missed our recall target
    :rtype: int

    """
    r_seen = df.seen_relevance.sum() # how many relevant docs have been seen
    urns = df[pd.notna(df.seen_relevance)].seen_relevance[::-1] # urns of previous 1,2,...,N documents
    urn_sizes = np.arange(urns.shape[0])+1 # The sizes of these urns
    # Now we calculate k_hat, which is the minimum number of documents there would have to be
    # in each of our urns for the urn to be in keeping with our null hypothesis
    # that we have missed our target
    k_hat = np.floor(
        r_seen/recall_target +1 - # We devide the number or relevant documents by our recall target and add 1
        (
            r_seen - # From this we subtract the total relevant documents seen
            urns.cumsum() # before each urn
        )
    )
    p = hypergeom.cdf( # the probability of observing
        urns.cumsum(), # the number of relevant documents in the sample
        df.shape[0] - (urns.shape[0] - urn_sizes), # In a population made up out of the urn and all remaining docs
        k_hat, # Where K_hat docs in the population are actually relevant
        urn_sizes # After observing this many documents
    )
    return min(p)
