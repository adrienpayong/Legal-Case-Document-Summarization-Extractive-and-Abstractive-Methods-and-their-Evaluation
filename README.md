# Review — Legal Case Document Summarization: Extractive and Abstractive Methods and their Evaluation
## Introduction

In Common Law systems, law practitioners have to read through hundreds of case judgements/rulings in order to identify relevant cases that they can cite as prece- dents in an ongoing case. This is a time-consuming process as case documents are generally very long and complex. Thus, automatic summarization of
legal case documents is an important problem The state-of-the-art in document summarization has advanced rapidly in the last couple of years, but there has not been much exploration on how these models perform on legal documents. Our prior work took an early step in this direction, but it mostly considered extractive methods.
## Some interesting insights

To bridge this gap, the researchers (1) develop three legal case judgement summarization datasets from case documents from the Indian and UK Supreme
Courts, and (2) reproduce/apply representative methods from several families of summarization models on these datasets, and analyse their performances
- **Domain-specific vs Domain-agnostic methods**: They apply several domain-independent summarization methodds including unsupervised extractive, supervised abstractive and supervised extractive and they reproduce several legal domain-specific summarization methods.
- **Domain-specific training/fine-tuning**: Using models pretrained on legal corpora, like Legal- Pegasus (leg), consistently improves performance.
- **How to deal with long documents**: A key challenge in using existing abstractive summarizers on legal documents is that the input capacity of such models is often much lower than the length of legal documents. Accordingly, the researchers experiment with three different approaches for summarizing long legal case documents – (i) applying long document summarizers such as Longformer (Beltagy et al., 2020) that are designed to handle long documents, (ii) applying short document summarizers such as BART (Lewis et al., 2020) and Legal-Pegasus (leg) together with approaches for chunking the docu ments, and (iii) reducing the size of the input document by first performing an extractive summarization and then going for abstractive summarization. In general, they find the chunking-based approach to perform better for legal documents, especially with fine-tuning, although Longformer performs the best on the UK-Abs dataset containing the longest documents, according to some of the metrics.
- **Evaluation of summary quality**: The researchers perform (i) document-wide automatic evaluations, (ii) segment-wise automatic evaluations, as well as (iii) evaluations by Law practitioners 
## Overview of Existing Summarization Algorithms
### Extractive domain-independent methods
Several domain-specific approaches have been specifically designed for summarizing legal case documents. Among unsupervised methods
- LetSum (Farzindar and Lapalme, 2004)
- KMM (Saravanan et al., 2006) rank sentences based on term distribution models (TF-IDF and k-mixture model respectively)
- CaseSummarizer (Polsley et al., 2016) ranks sentences based on their TF-IDF weights coupled with legal-specific features
- MMR (Zhong et al., 2019) generates a template-based summary using a 2-stage classifier and a Maximum Margin Relevance (Zhong et al., 2019) module
- Gist (Liu and Chen, 2019) is the only supervised method specifically designed for summarizing legal case documents. Gist first represents a sentence with different handcrafted features. It then uses 3 models – MLP, Gradient Boosted Decision Tree, and LSTM to rank sentences in order of their likelihood to be included in the summary. 

### Abstractive methods
Most abstractive summarization models have an input token limit which is usually shorter than the length of legal case documents. Approaches from this family include:
- Pointer-Generator (See et al., 2017)
- BERTSumAbs (Liu and Lapata, 2019)
- Pegasus (Zhang et al 2020)
- BART (Lewis et al., 2019)
- Models like Longformer (Beltagy et al., 2020) introduce transformer architectures with more efficient atten- tion mechanisms that enables them to summarize
long documents (up to 16 × 1024 input tokens)
- the only method for abstractive legal document summarization is LegalSumm (Feijo and Moreira, 2021). The method uses the RulingBR dataset (in Portuguese language) which has much shorter documents and summaries than the datasets in this work.

## Datasets for Legal Summarization
There are very few publicly available datasets for legal case document summarization, especially in English. The researchers develop the following three datasets:
- Indian-Abstractive dataset (IN-Abs):The researchers collect Indian Supreme Court judgements from the website of Legal Information Institute of
India (http://www.liiofindia.org/in/cases/cen/INSC/) which provides free and non-profit access to databases of Indian law.
- Indian-Extractive dataset (IN-Ext): 
- UK-Abstractive dataset (UK-Abs): The UK Supreme court website (https://www.supremecourt.uk/decided-cases/) provides all cases judgements that were ruled since the year 2009. For most of the cases, along with the judgements, they also provide the official press
summaries of the cases, which we consider as the reference summary.
- Table 2 provides a summary of the datasets, while Table 1 compares the length of the documents in these datasets with those in other datasets. Note that the documents in UK-Abs are approximately double the length of the IN-Abs and IN-Ext documents, and have a very low compression ratio (0.11); hence the UK-Abs dataset is the most challenging one for automatic summarization.

## Experimental Setup and Evaluation
- **Target length of summaries**: During inference, the trained summarization models need to be provided with the target length of summaries L (in number of words). For every document in the IN-Ext dataset, we have two reference summaries(written by two experts). Each model is made to gen-erate a summary of length at most L words.
- **Evaluation of summary quality**: The rsearchers report ROUGE-1, ROUGE-2, and ROUGE-L F-scores (computed using https://pypi.org/
project/py-rouge/, with max_n set to 2, parameters limit_length and length_limit not used, and other parameters kept as default), and BertScore (Zhang et al., 2019) (computed using https://pypi.org/project/ bert-score/ version 0.3.4) that calculates the semantic similarity scores using the pretrained
BERT model. They calculate two kinds of ROUGE and BERTScore as follows: 

(a) Overall document-wide scores: For a given document, the researchers compute the ROUGE and BERTScore of an algorithmic summary with respect to the reference summary. For IN-Ext, they compute the scores individually with each of the two reference summaries and take the average. The scores are averaged over all documents in the evaluation set.

(b) Segment-wise scores: In legal case judgement summarization, a segment-wise evaluation is important to understand how well each rhetorical segment has been summarized (Bhattacharya et al.,2019). We can perform this evaluation only for the IN-Ext and UK-Abs datasets (and not for IN-Abs), where the reference summaries are written segment-wise. 

- **Expert evaluation**: The researchers select a few methods (that achieve the highest ROUGE scores) and get the summaries generated by them for a few documents evaluated by three Law experts
- **Consistency scores**: It is important to measure the consistency of an algorithmic summary with the original document, given the possibility of hallucination by abstractive models.

## Extractive Summarization Methods
The researchers consider some representative methods from four classes of extractive summarizers: 

(1) Legal domain-specific unsupervised methods: 

- LetSum
- KMM
- CaseSummarizer
- MMR. 

(2) Legal domain-specific supervised methods: 
- Gist

(3) Domain-independent unsupervised methods:
- LexRank
- LSA
- DSDR
- Luhn
- Reduction
- PacSum. 

(4) Domain-independent supervised methods:
- SummaRuNNer
- BERTSum.

**Training supervised extractive models**:
- he supervised methods (Gist, SummaRuNNer and BERTSUM) require labelled training data, where every sentence must be labeled as 1 if the sentence is suitable for inclusion in the summary, and 0 otherwise.
- The researchers explore three methods – Maximal, Avr, and TF-IDF – for converting the abstractive summaries to their extractive counterparts. Best performances for the supervised methods are observed when the training data is generated through the Avr method.

## Abstractive Summarization Methods
**Models meant for short documents**: The researchers consider Legal-Pegasus (leg) which is already pretrained on legal documents, and BART (Lewis et al., 2020) (max input length of 1024 tokens). They use their pre-trained versions from the HuggingFace library.

**Chunking-based approach**: They first divide a document into small chunks, the size of each chunk being the maximum number of tokens (say, n) that a model is designed/pre-trained to accept without truncating (e.g., n = 1024 for BART). Specifically, the first n tokens (without breaking sentences) go to the first chunk, the next n tokens go to the second chunk, and so on. Then we use a model to summarize every chunk. For a given document, we
equally divide the target summary length among all the chunks. Finally, we append the generated summaries for each chunk in sequence.

**Models meant for long documents**: Models like Longformer (LED) (Beltagy et al., 2020) have been especially designed to handle long documents (input capacity = 16,384 tokens), by in- cluding an attention mechanism that scales linearly with sequence length. We use Legal-LED specifi- cally finetuned on legal data. The model could accommodate most case documents fully. 

**Hybrid extractive-abstractive approach**: First, the document length is reduced by selecting salient sentences using a BERT-based extractive summarization model. Then a BART model is used to generate the final summary (Bajaj et al., 2021). Since, in our case, we often require a summary length greater than
1024, we use a chunking-based BART (rather than pre-trained BART) in the second step. The researchers call this model BERT_BART.

## Generating finetuning data

Finetuning supervised models needs a large set of doc-summary pairs. However, the considered models (apart from Longformer) have a restricted input limit which is lesser than the length of documents in our datasets. Hence, the researchers use the following method, inspired from Gidiotis and Tsoumakas (2020), to generate finetuning data for chunking based summarization.

- Consider (d, s) to be a (training document, reference summary) pair. When d is segmented into n chunks d1, d2, ... dn, it is not logical for the same
s to be the reference summary for each chunk di.
- To generate a suitable reference summary si for each chunk di, first we map every sentence in s to the most similar sentence in d.
- For every chunk di, we combine all sentences in s which are mapped to any of the sentences in di, and consider those sentences as
the summary si (of di). 
- Following this procedure from each document, we get a large number of (di, si) pairs which are then used for finetuning.

## Sentence similarity measures for generating fine-tuning data
  The researchers experiment with several techniques for measuring sentence similarity between two sentences:
  - Mean Cosine Similarity (MCS)
  - Smooth Inverse Frequency (SIF)
  - Cosine similarity between BERT [CLS] token embeddings (CLS)
  - MCS_RR which incorporates rhetorical role information. 
  
Out of these, the researchers find MCS to perform the best.

## Evaluation of Extractive methods

**Overall Evaluation (Tables 3–5)**: Among the unsupervised general methods, Luhn (on IN-Ext) and DSDR (on IN-Abs and UK-Abs) show the best per-
formances. Among the unsupervised legal-specific methods, CaseSummarizer performs the best on both In-Abs and UK-Abs datasets, while LetSum performs the best on IN-Ext. Among supervised extractive methods, SummaRuNNer performs the best across both domain-independent and domain-specific categories, on the IN-Abs and UK-Abs datasets. BERT-Ext is the best performing model on the IN-Ext dataset.

**Segment-wise Evaluation (Tables 6, 7)**: None of the methods performs well across all segments, and fine-tuning generally improves perfor-
Table 6 and Table 7 show the segment-wise ROUGE-L Recall scores of some of the best performing methods on the IN-Ext
and UK-Abs datasets respectively.

## Evaluation of Abstractive methods

**Overall Evaluation (Tables 3–5)**: Among the pretrained models, Legal-Pegasus generates the best summaries (Table 3), followed by BART-based methods. This is expected, since Legal-Pegasus is pre-trained on legal documents.This short document summarizer, when used with chunking to handle long documents, notably outperforms Legal- LED, which is meant for long documents. For IN-Ext dataset, BERT_BART performs the best maybe
due to extractive nature of the summaries. All models show notable improvement through fine-tuning. Overall, the best performances are
noted by Legal-Pegasus (IN-Ext and IN-Abs) and BART_MCS (UK-Abs).

**Segment-wise Evaluation (Tables 6, 7)**: None of the methods performs well across all segments, and fine-tuning generally improves perfor-
mance. Interestingly, though Legal-LED performs poorly with respect to document-wide ROUGE scores, it shows better performance in segment-wise evaluation – it gives the best performance in the FAC and ARG segments of IN-Ext and in 2 out of the 3 segments of UK-Abs.
Overall performance on long legal case documents:



# Citation
If you are using the implementations, please refer to the following papers:
```
@inproceedings{bhattacharya2021,
  title={Legal Case Document Summarization: Extractive and Abstractive Methods and their Evaluation},
  author={Shukla, Abhay and Bhattacharya, Paheli and Poddar, Soham and Mukherjee, Rajdeep and Ghosh, Kripabandhu and Goyal, Pawan and Ghosh, Saptarshi},
  booktitle={The 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing},
  year={2022}
}

@inproceedings{bhattacharya2019comparative,
  title={A comparative study of summarization algorithms applied to legal case judgments},
  author={Bhattacharya, Paheli and Hiware, Kaustubh and Rajgaria, Subham and Pochhi, Nilay and Ghosh, Kripabandhu and Ghosh, Saptarshi},
  booktitle={European Conference on Information Retrieval},
  pages={413--428},
  year={2019},
  organization={Springer}
}
```

