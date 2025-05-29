Bitcoin-Alpha, email-Enron, email-EuAll, soc-Epinions1, soc-Pokec can be downloaded from [SNAP](https://snap.stanford.edu/data/index.html).

> Leskovec and A. Krevl. SNAP Datasets: Stanford large network dataset collection, 2014.

hiv-Trans is obtained from [HIV Transmission Network Metastudy Project: An Archive of Data From Eight Network Studies,1988--2001 (ICPSR 22140)](https://www.icpsr.umich.edu/web/NAHDAP/studies/22140)

Some statsics of the datasets are shown below:

| **Domain**          | **Dataset**        | **#Nodes**  | **#Edges**   | **Avg Degree** |
|---------------------|-------------------|------------|------------|---------------|
| **Bitcoin Trade**  | Bitcoin-Alpha     | 3,783      | 24,186     | 6.39          |
| **Communication**  | email-Enron       | 36,692     | 367,662    | 10.02         |
|                    | email-EuAll       | 265,214    | 420,045    | 1.58          |
| **Social**        | soc-Epinions     | 75,879     | 508,837    | 6.71          |
|                    | soc-Pokec         | 1,632,803  | 30,622,564 | 18.75         |

Nodes with no in-degrees or out-degrees (dangling nodes) are removed, the remaining nodes are relabeled, and undirected edges are represented by two directed edges.

