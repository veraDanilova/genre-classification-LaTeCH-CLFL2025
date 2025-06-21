# genre-classification-LaTeCH-CLFL2025

Code for the paper **"Classifying Textual Genre in Historical Magazines (1875-1990)"** (LaTeCH-CLFL2025)

## Project Description

This repository is part of the ERC-funded project **ActDisease** (ERC-2021-STG 10104099), a research initiative in the modern history of medicine. The project studies the history of historical medical magazines issued by European patient organizations throughout the 20th century.

The **ActDisease3 Dataset** is an extensive private collection currently being digitized as part of this ongoing project. It consists of medical magazines issued by ten European patient organizations in four languages (Swedish, German, French, and English) throughout the 20th century. Each magazine had a different publication frequency, resulting in a varying number of issues per year and featuring diverse page formats and visually complex layouts. Within the same page, these magazines often combine texts that carry different communicative purposes, such as personal narratives, advertisements, instructions, short stories, etc.

**Note:** Due to privacy concerns, we cannot share this dataset publicly. Researchers interested in accessing the data should contact the author of this repo.

## Repository Structure

- `classification/` – Scripts for dataset preparation and genre classification
- `few-shot-experiment/` – Scripts and data for few-shot and semi-supervised experiments
- `zero-shot-experiment/` – Scripts and data for zero-shot experiments
- `lib/` – Utility functions
- `plots/` – Plots and visualizations of results
- `requirements.txt` – Python dependencies
- `few_shot_test.ipynb`, `zero_shot_test.ipynb` – Notebooks for evaluation on test data and analyzing results


## Reference
If you use this code or data in your research, please cite the following paper:

```bash
@inproceedings{danilova-soderfeldt-2025-classifying,
    title = "Classifying Textual Genre in Historical Magazines (1875-1990)",
    author = {Danilova, Vera  and
      S{\"o}derfeldt, Ylva},
    editor = "Kazantseva, Anna  and
      Szpakowicz, Stan  and
      Degaetano-Ortlieb, Stefania  and
      Bizzoni, Yuri  and
      Pagel, Janis",
    booktitle = "Proceedings of the 9th Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature (LaTeCH-CLfL 2025)",
    month = may,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.latechclfl-1.15/",
    doi = "10.18653/v1/2025.latechclfl-1.15",
    pages = "160--171",
    ISBN = "979-8-89176-241-1",
    abstract = "Historical magazines are a valuable resource for understanding the past, offering insights into everyday life, culture, and evolving social attitudes. They often feature diverse layouts and genres. Short stories, guides, announcements, and promotions can all appear side by side on the same page. Without grouping these documents by genre, term counts and topic models may lead to incorrect interpretations.This study takes a step towards addressing this issue by focusing on genre classification within a digitized collection of European medical magazines in Swedish and German. We explore 2 scenarios: 1) leveraging the available web genre datasets for zero-shot genre prediction, 2) semi-supervised learning over the few-shot setup. This paper offers the first experimental insights in this direction.We find that 1) with a custom genre scheme tailored to historical dataset characteristics it is possible to effectively utilize categories from web genre datasets for cross-domain and cross-lingual zero-shot prediction, 2) semi-supervised training gives considerable advantages over few-shot for all models, particularly for the historical multilingual BERT."
}
```