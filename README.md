
# Content-based and Collaborative Filtering Recommender Systems
Katherine Huerta

## Getting Started (Overall)
### Prerequisites
* Python 3.7 (python >3.5= should work as well)
* pandas
* numpy
* matplotlib
* jupyter notebook

#### Optional

I used Anaconda as my platform. It provides Jupyter Notebook and allows you to install packages easily. Instructions on installation are provided below:

[Anaconda Installation on Windows](https://docs.anaconda.com/anaconda/install/windows/)

[Anaconda Installation on macOS](https://docs.anaconda.com/anaconda/install/mac-os/)

[Anaconda Installation on Linux](https://docs.anaconda.com/anaconda/install/linux/)

Additional information for getting started with Python and Anaconda can be found [here](https://docs.anaconda.com/anaconda/user-guide/getting-started/)

# Part 1: Content-Based Filtering

In the first notebook, we create two **content-based** movie recommenders. The first recommender is based on the plot description. The user can type in a movie they enjoyed, and the recommendations will be based on similarities to the plot description of the input movie.

The second recommender is based on credits, genres, and keywords. The user will provide a movie as an input, and the resulting recommendations will be based on movies with similar genre, actors, director, and keywords. 

We compute similarity scores by first computing Term Frequency-Inverse Document Frequency (TF-IDF) vectors. The dot product of this matrix yields the cosine similarity score.

We then define a function that takes a movie title as an input, where the output is a list of the ten most similar movies. 

Finally, we test it out to see what our recommendations are!

## Getting Started

### Prerequisites
* scikit-learn
    * sklearn:
        * TfidfVectorizer
        * CountVectorizer
        * linear_kernel
        * cosine_similarity
* ast: literal_eval

Ast should already be part of your environment. You can install scikit-learn following these [instructions](https://scikit-learn.org/stable/install.html). If you are using the anaconda navigator as suggested, simply paste this code into your anaconda prompt:
```
conda install scikit-learn
```


#### Code for importing packages

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
```
#### Download the datasets, notebook, and image
1. Download my folder (Huerta_Katherine_Project) and extract the folder
2. From the folder **Part 1**, upload the following into your jupyter notebook environment
    * upload tmdb_5000_credits.csv and tmdb_5000_movies.csv from the **tmdb-movie-metadata** folder.
    * upload the content_based_image from the folder as well (if you want to view the image)
    * upload the notebook file (Huerta_Part1_ContentBased.ipnyb)
    * **Note** Keep the file names as they are, so the code works without modification. Make sure all files are in the same folder that the notebook is in.
3. Read in the datasets using pandas. The code example below is already included in my notebook, so no need to do this separately
```python
df1 = pd.read_csv('tmdb_5000_credits.csv')
df2 = pd.read_csv('tmdb_5000_movies.csv')
```

## Datasets
* Using the [TMDB 5000 Movie Dataset](https://www.kaggle.com/tmdb/tmdb-movie-metadata) from Kaggle.
    * This dataset contains metadata on approximately 5,000 movies from TMDb. This makes it an ideal dataset for a content-based recommendation system.
   1. tmdb_5000_credits.csv
   2. tmdb_5000_movies.csv
    * Both datasets are located in the tmdb-movie-metadata folder (\Huerta_Katherine_Project\Part 1)
    * More information on these datasets can be found in my jupyter notebook file (Huerta_Part1_ContentBased), or using the link above. 

# Acknowledgments
1. [TMDB 5000 Movie Dataset](https://www.kaggle.com/tmdb/tmdb-movie-metadata) Metadata on ~5,000 movies from TMDb
2. [Ibtesam Ahmed](https://www.kaggle.com/ibtesama/getting-started-with-a-movie-recommendation-system) Getting Started with a Movie Recommendation System
3. Image link for introduction: [Emma Grimaldi](https://towardsdatascience.com/how-to-build-from-scratch-a-content-based-movie-recommender-with-natural-language-processing-25ad400eb243)

# Part 2: Collaborative Filtering Recommender Systems

The idea behind this recommendation engine is to attempt to predict the rating that a user would give a movie. The dataset we are using is a list of movies, users, and user ratings with links to tmdb (theMovieDB) web IDs. This allows us to obtain information from APIs related to our recommendations.

Essentially, we will build a matrix of users X movies with their rating values filled in for existing data. This feature set is decomposed into UxV matrices that relate to a set of latent factors that are used to describe user preferences, and ultimately their expected ratings.

Apache Spark ML uses alternating least squares (ALS) for collaborative filtering, and it is a common algorithm used for recommenders. An ALS recommender is a matrix factorization algorithm that uses ALS with Weighted-Lambda-Regularization (ALS-WR). This reduces the dependency of the regularization paramater on the scale of the dataset. Thus, the best parameter learned from a sample subset is applicable to the entire dataset with similar performance. The latent factors explain the user of interest's movie ratings and map new users to optimized movie recommendations. 

## Getting Started
Most of these packages' imports are included in the notebook, but some are not immediately available and will need more instruction/preparation before running the code. If you already have Apache Spark and pyspark, you should be fine to run **most** of the code. **However, see tmdbsimple bullet under prerequesites to ensure you can display the movie poster visualization** 
### Prerequisites
* math
* os, sys, requests, json
* time
* Image
* display
* Apache Spark
    1. Must have [Java 8](https://docs.oracle.com/javase/8/docs/technotes/guides/install/install_overview.html) or higher installed
    2. Go to [Spark downloads page](http://spark.apache.org/downloads.html), select latest Spark release (prebuilt package for Hadoop), download it directly, and unzip and move it to the appropriate folder (video on how to set up spark can be found [here](https://www.youtube.com/watch?v=VYNsaR-gOsA), and another helpful article that includes pyspark implementation can be found [here](https://www.sicara.ai/blog/2017-05-02-get-started-pyspark-jupyter-notebook-3-minutes) - highly suggest you watch if you don't have it installed already).

* pyspark
    1. Once you have Spark set up, there are many ways to configure pyspark with jupyter notebook: 
        * Anaconda has its own pyspark package. You can simply search for pyspark in your anaconda environment (make sure you check the "not installed" option first) and install it.
        * More installation options can be found in [Anaconda Cloud Packages](https://anaconda.org/conda-forge/findspark)
        * You can install conda findspark to access spark instance from jupyter notebook, enabling pyspark to be importable. 
            * Conda prompt:
            ``` 
            conda install -c conda-forge findspark
            ```
            * Then open Jupyter notebook and input the following code:
            ```
            import findspark
            findspark.init()
            findspark.find()
            import pyspark
            ```
        * You can explore the pyspark package API in Apache Spark's [PySpark 2.4.4 documentation](https://spark.apache.org/docs/latest/api/python/pyspark.html)
    2. Packages from pyspark used in this notebook:
        * functions
        * DataFrameNaFunctions
        * udf, col, when
        * RegressionEvaluator
        * ALS
        * CrossValidator
        * ParamGridBuilder
        * Pipeline
        * Row
        * SQLContext
       
* **tmdbsimple** (for displaying movie posters)
    1. Make an account on [themoviedb.org](https://www.themoviedb.org/). Instructions can be found [here](https://developers.themoviedb.org/3/getting-started/introduction).
        * **Note:** Please use your own API key. 
    2. To install tmdbsimple:
        * If you are using the anaconda platform, you can add it to your environment following instructions provided by [Anaconda Cloud](https://anaconda.org/anjos/tmdbsimple). Simply use the anaconda (not python) prompt to input this code for installation of this package:
        ```
        conda install -c anjos tmdbsimple
        ```
        * Instructions on how to install using pip can be found [here](https://pypi.org/project/tmdbsimple/)
        
#### Example Code for Prerequisites:
This is already included in the notebook, but just in case you wanted to know which pyspark subpackages are being used, or how the tmdb API key will be used, see the following:

```python
from pyspark.sql import functions as F
from pyspark.sql import DataFrameNaFunctions as DFna
from pyspark.sql.functions import udf, col, when
import matplotlib.pyplot as plt
import pyspark as ps
import os, sys, requests, json

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.sql import Row
import numpy as np
import math

from pyspark.sql import SQLContext
import time
import requests

import tmdbsimple as tmdb
tmdb.api_key = 'Your_API_Key'

from IPython.display import Image
from IPython.display import display

import pandas as pd
```

#### Download the datasets, notebook, and image

1. Download my folder (Huerta_Katherine_Project) and extract/unzip the folder
2. From the folder **Part 2**, upload the following into your jupyter notebook environment
    * upload files from the **ml-latest-small** folder to jupyter notebook.
    * upload the two images from the images folder as well
    * upload the notebook file (Huerta_Part2_CollaborativeFiltering_SparkALS.ipnyb)
    * **Note** Keep the file names as they are, so the code works without modification. Make sure all files are in the same folder that the notebook is in.
3. Read in the datasets using `spark.read.csv`. The code example below is already included in my notebook. Doing this prematurely will result in failure because you need to start a SparkSession and create a Spark Context.
```python
# read movies CSV
movies_df = spark.read.csv('movies.csv',
                         header=True,       # use headers
                         quote='"',         # char for quotes
                         sep=",",           # char for separation
                         inferSchema=True)  # Infer Schema
movies_df.printSchema()
```

```python
# read ratings CSV
ratings_df = spark.read.csv('ratings.csv',
                         header=True,       # use headers
                         quote='"',         # char for quotes
                         sep=",",           # char for separation
                         inferSchema=True)  # infer schema
ratings_df.printSchema()
```

```python
# read links CSV
links_df = spark.read.csv('links.csv',
                         header=True,       # use headers
                         quote='"',         # char for quotes
                         sep=",",           # char for separation
                         inferSchema=True)  # infer schema
links_df.printSchema()
```


## Datasets
* Using the MovieLens dataset provided by [GroupLens](https://grouplens.org/datasets/movielens/)
* Specifically, using [MovieLens Latest Datasets](https://grouplens.org/datasets/movielens/latest/)
    * For simulation, use **ml-latest-small.zip** (size: 1 MB)
        * _Small_: ~100,000 movie ratings from users (on a 1-5 scale) and ~3,600 tag applications applied to ~9,000 movies by ~600 users (last updated in 2018).
    * Also works for **ml-latest.zip**, but is to large to submit on blackboard (size: 265 MB).
        * _Full_: 27,000,000 ratings and 1,100,000 tag applications applied to 58,000 movies by 280,000 users (2018).
* The datasets can be found in the **ml-latest-small** folder (\Huerta_Katherine_Project\Part 2) and contains:
    * `ratings.csv` 
        * All ratings contained in this file
        * Each line of this file after the header row represents one rating of one movie by one user
        * File format:
        ``` userId,movieId,rating,timestamp ```
    * `tags.csv`
        * All tags contained in this file
        * Each line of this file after the header row represents one tag applied to one movie by one user
        * File format:
        ```userId,movieId,tag,timestamp```
    * `movies.csv`
        * Movies data file
        * Contains movie information
        * Each line of file after header represents one movie
        * File format:
        ```movieId,title,genres```
    * `links.csv`
        * Links data file
        * Contains identifiers that can be used to link to other sources of movie data.
        * Each line of file after header row represents one movie
        * File format:
        ```movieId,imdbId,tmdbId```
        * **used for displaying poster images**
            * movieId - identifier for movies used by <https://movielens.org>
                * Ex: Movie "Toy Story" has link <https://movielens.org/movies/1> 
            * imdbId - identifier for movies used by <http://www.imdb.com>
            * **tmdbId** - identifier for movies used by <https://www.themoviedb.org>.
                * Toy Story has the link <https://www.themoviedb.org/movie/862>.
                * **identifier used in this system**



### About:
* **userId**: 
    * User Ids (integer)
    * randomly selected users. User ids are consistent between `ratings.csv` and `tags.csv` (i.e., the same id refers to the same user across the two files).
* **movieId**:
    * Movie Ids
    * Only movies with at least one rating or tag are included in the dataset. These movie ids are consistent with those used on the MovieLens web site (id `1` corresponds to the URL <https://movielens.org/movies/1>). Movie ids are consistent between `ratings.csv`, `tags.csv`, `movies.csv`, and `links.csv` (the same id refers to the same movie across these four data files).
* **rating**:
    * User ratings
    * Ratings are on a 5-star scale, with half-star increments (0.5 stars - 5.0 stars).
* **timestamp**: seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970.

* **tag**:
    * user-generated metadata about movies. 
    * Each tag is usually a single word or short phrase. The meaning, value, and purpose of a particular tag is determined by each user.
* **title**:
    * Movie titles entered manually or imported from <https://www.themoviedb.org/>, and include the year of release in parentheses.
* **genres**:Pipe-separated list. Selected from:

    * Action
    * Adventure
    * Animation
    * Children's
    * Comedy
    * Crime
    * Documentary
    * Drama
    * Fantasy
    * Film-Noir
    * Horror
    * Musical
    * Mystery
    * Romance
    * Sci-Fi
    * Thriller
    * War
    * Western
    * (no genres listed)

## Acknowledgments
1. [GdMacmillan - spark_recommender_systems](https://github.com/GdMacmillan/spark_recommender_systems/blob/master/spark_recommender.ipynb) backbone of my code, however I had to make many modification for it to work for my environment/dataset
2. [Apache Spark - Collaborative Filtering](https://spark.apache.org/docs/2.2.0/ml-collaborative-filtering.html) 
3. [tmdbsimple 2.2.0](https://pypi.org/project/tmdbsimple/) _A Python wrapper for The Movie Database API v3_
4. [The Movie DB](https://www.themoviedb.org/) for displaying pictures of movies. 
5. [GroupLens](https://grouplens.org/datasets/movielens/) [MovieLens](https://movielens.org/) dataset source. I used the ml-latest-small (most recent dataset containing 100,000 ratings, 3,600 tag applications applied to 9,000 movies by 600 users). I used the small dataset to provide a faster simulation (size: 1 MB).
6. [Mark Litwintschik](https://tech.marksblogg.com/recommendation-engine-spark-python.html) _Recommendation Engine built using Spark and Python_ Explains how a user can get predictions based on their own ratings of movies.
7. [Abhinav Ajitsaria](https://realpython.com/build-recommendation-engine-collaborative-filtering/) Rating Matrix image used in introduction.
8. [Kevin Liao](https://towardsdatascience.com/prototyping-a-recommender-system-step-by-step-part-2-alternating-least-square-als-matrix-4a76c58714a1) Prototyping a Recommender System Step by Step Part 2: Alternating Least Square (ALS) Matrix Factorization in Collaborative Filtering. _Used this article to learn more about ALS and Matrix Factorization_
