import math
from scipy.stats import hypergeom, nchypergeom_wallenius
import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import List, Union
import matplotlib.pyplot as plt

Array = np.ndarray[tuple[int], np.dtype[np.int_]]
ListLike = Union[Array, List[int], pd.Series]

#
# def k_h0(r_al, r_seen, recall_target):
#     """
#     Calculate the smallest number of relevant documents there would have to be in our urn, for us to have missed a given recall recall_target
#
#     :param r_seen: the number of relevant documents seen.
#     :type kind: int
#     :return: The number smallest number of relevant documents that could be in our urn that there could be if we had missed our recall target
#     :rtype: int
#
#     """
#     return math.floor(r_seen/recall_target-r_al+1)


def generate_dataset(
    N: int=20000,
    prevalence: float=0.01,
    bias: float=10,
    random_seed: Union[int, None]=None
    ) -> pd.DataFrame:
    """
    Generate a dataset resembling the kind created through machine learning
    prioritised screening.

    :param N: The number of documents returned by the query
    :type N: int
    :param prevalence: The proportion of those documents which are relevant
    :type prevalence: float
    :param bias: The likelihood of drawing a random relevant document over the
        likelihood of drawing a random irrelevant document. The higher this is,
        the better our ML has worked.
    :type bias: float
    :param random_seed: A random seed. Set this to ensure the same sequence of
        documents is drawn each time the code is run.
    :type random_seed: int|None
    :return: A dataframe with a `N` rows, of which `prevalence`*`N` are relevant.
        The column `relevant` is made up of 1s and 0s, where 1 represents a
        relevant, and 0 an irrelevant document
    :rtype: pd.DataFrame

    """
    sample = np.zeros(N, dtype=np.int8)
    n_relevant_docs = round(N*prevalence)
    sample[:n_relevant_docs] = 1
    ps = np.ones(N)
    ps[:n_relevant_docs] = bias
    ps = ps/ps.sum()

    if random_seed is not None:
        rng = np.random.default_rng(seed=random_seed)
        sample = rng.choice(sample,sample.shape,replace=False, p=ps)
    else:
        sample = np.random.choice(sample,sample.shape,replace=False, p=ps)
    df = pd.DataFrame({'relevant': sample})
    df['id'] = df.index
    return df

def calculate_h0(
    labels_: ListLike,
    N: int,
    recall_target: float=.95,
    bias: float=1,
    ) -> float | None:
    """
    Calculates a p-score for our null hypothesis h0, that we have missed our recall target `recall_target`.

    :param labels_: An ordered sequence of 1s and 0s representing, in the order
        in which they were screened, relevant and irrelevant documents
        respectively.
    :type labels_: list|np.array|pd.Series
    :param N: The total number of documents from which you want to find the
        relevant examples. The size of the haystack.
    :type N: int
    :param recall_target: The proportion of truly relevant documents you want
        to find, defaults to 0.95
    :type recall_target: float
    :param bias: The assumed likelihood of drawing a random relevant document
        over the likelihood of drawing a random irrelevant document. The higher
        this is, the better our ML has worked. When this is different to 1,
        we calculate the p score using biased urns.
    :type bias: float
    :return: a p-score for our null hypothesis. We can reject the null hypothesis (and stop screening) if p is below 1 - our confidence level.
    :rtype: float

    """
    labels: npt.NDArray[np.int_] = (labels_ if type(labels_) is np.ndarray
                                else np.array(labels_, dtype=np.int8))

    r_seen = labels.sum() # how many relevant docs have been seen
    urns = labels[::-1] # urns of previous 1,2,...,N documents
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
    if bias == 1:
        p = hypergeom.cdf( # the probability of observing
            urns.cumsum(), # the number of relevant documents in the sample
            N - (urns.shape[0] - urn_sizes), # In a population made up out of the urn and all remaining docs
            k_hat, # Where K_hat docs in the population are actually relevant
            urn_sizes # After observing this many documents
        )
    else:
        p = nchypergeom_wallenius.cdf(
            urns.cumsum(), # the number of relevant documents in the sample
            N - (urns.shape[0] - urn_sizes), # In a population made up out of the urn and all remaining docs
            k_hat, # Where K_hat docs in the population are actually relevant
            urn_sizes, # After observing this many documents
            bias # Where we are bias times more likely to pick a random relevant document
        )
    return min(p)

def recall_frontier(
    labels_: ListLike,
    N: int,
    bias: float=1,
    plot: bool=True
) -> dict:
    """
    Calculates a p-score for our null hypothesis h0, that we have missed our recall target `recall_target`, across a range of recall_targets.

    :param labels_: An ordered sequence of 1s and 0s representing, in the order
        in which they were screened, relevant and irrelevant documents
        respectively.
    :type labels_: list|np.array|pd.Series
    :param N: The total number of documents from which you want to find the
        relevant examples. The size of the haystack.
    :type N: int
    :param bias: The assumed likelihood of drawing a random relevant document
        over the likelihood of drawing a random irrelevant document. The higher
        this is, the better our ML has worked. When this is different to 1,
        we calculate the p score using biased urns.
    :type bias: float
    :return: A dictionary containing a list of recall targets: `recall_target`.
        alongside a list of p-scores: `p`.
    :rtype: dict

    """

    recall_target = 0.99
    p_scores = []
    recall_targets = []
    while recall_target > 0:
        p = calculate_h0(labels_, N, recall_target, bias)
        p_scores.append(p)
        recall_targets.append(recall_target)
        recall_target-=0.005
        if p < 0.01:
            break

    if plot:
        plt.plot(recall_targets, p_scores, 'o-')
    return {'recall_target': recall_targets, 'p': p_scores}

def retrospective_h0(
    labels_: ListLike,
    N: int,
    recall_target: float=0.95,
    bias: float=1,
    batch_size: int=1000,
    confidence_level: float=0.95,
    plot: bool=True
) -> dict:
    """
    Calculates a p-score for our null hypothesis h0, that we have missed our recall target `recall_target`, every `batch_size` documents

    :param labels_: An ordered sequence of 1s and 0s representing, in the order
        in which they were screened, relevant and irrelevant documents
        respectively.
    :type labels_: list|np.array|pd.Series
    :param N: The total number of documents from which you want to find the
        relevant examples. The size of the haystack.
    :type N: int
    :param recall_target: The proportion of truly relevant documents you want
        to find, defaults to 0.95
    :type recall_target: float
    :param bias: The assumed likelihood of drawing a random relevant document
        over the likelihood of drawing a random irrelevant document. The higher
        this is, the better our ML has worked. When this is different to 1,
        we calculate the p score using biased urns.
    :type bias: float
    :param batch_size: The size of the batches for which we will calculate our
        stopping criteria. Smaller batches = greater granularity = more
        computation time.
    :type batch_size: int
    :param confidence_level: The score will be calculated until p is smaller
        than 1-`confidence_level`
    :type batch_size: int
    :param plot: Whether to do a plot
    :type plot: bool
    :return: A dictionary containing a list of batch sizes: `batch_sizes`.
        alongside a list of p-scores: `p`.
    :rtype: dict

    """

    labels: npt.NDArray[np.int_] = (labels_ if type(labels_) is np.ndarray
                                else np.array(labels_, dtype=np.int8))

    batch_sizes = np.arange(labels.shape[0])[batch_size::batch_size]
    batch_sizes = np.append(batch_sizes, labels.shape[0])

    batch_ps = []

    for n_seen_batch in batch_sizes:
        batch_labels = labels[:n_seen_batch]
        p_h0 = calculate_h0(batch_labels, N=N, recall_target=recall_target, bias=bias)
        batch_ps.append(p_h0)

        if p_h0 is not None and p_h0 < (1.0 - confidence_level):
            break

    batch_ps = np.array(batch_ps)
    batch_sizes = batch_sizes[:batch_ps.shape[0]]


    if plot:
        fig, ax = plt.subplots()
        ax.plot(labels.cumsum())
        ax2 = ax.twinx()
        ax2.scatter(batch_sizes, batch_ps)
        ax.set_xlabel('Documents screened')
        ax.set_ylabel('Relevant documents identified')
        ax2.set_ylabel('p score for H0 that recall target missed')
    return {'batch_sizes': batch_sizes, 'p': batch_ps}
