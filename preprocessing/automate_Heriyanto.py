{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Heriyan-LA/Eksperimen_SML_Heriyanto/blob/main/preprocessing/automate_Heriyanto.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **1. Perkenalan Dataset**\n"
      ],
      "metadata": {
        "id": "kZLRMFl0JyyQ"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Tahap pertama, Anda harus mencari dan menggunakan dataset dengan ketentuan sebagai berikut:\n",
        "\n",
        "1. **Sumber Dataset**:  \n",
        "   Dataset dapat diperoleh dari berbagai sumber, seperti public repositories (*Kaggle*, *UCI ML Repository*, *Open Data*) atau data primer yang Anda kumpulkan sendiri.\n"
      ],
      "metadata": {
        "id": "hssSDn-5n3HR"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **2. Import Library**"
      ],
      "metadata": {
        "id": "fKADPWcFKlj3"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Pada tahap ini, Anda perlu mengimpor beberapa pustaka (library) Python yang dibutuhkan untuk analisis data dan pembangunan model machine learning atau deep learning."
      ],
      "metadata": {
        "id": "LgA3ERnVn84N"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install seaborn"
      ],
      "metadata": {
        "id": "BlmvjLY9M4Yj",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "collapsed": true,
        "outputId": "1ab731b9-5796-4d86-d18a-0262e43ea840"
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: seaborn in /usr/local/lib/python3.11/dist-packages (0.13.2)\n",
            "Requirement already satisfied: numpy!=1.24.0,>=1.20 in /usr/local/lib/python3.11/dist-packages (from seaborn) (2.0.2)\n",
            "Requirement already satisfied: pandas>=1.2 in /usr/local/lib/python3.11/dist-packages (from seaborn) (2.2.2)\n",
            "Requirement already satisfied: matplotlib!=3.6.1,>=3.4 in /usr/local/lib/python3.11/dist-packages (from seaborn) (3.10.0)\n",
            "Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (1.3.2)\n",
            "Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.11/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (0.12.1)\n",
            "Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.11/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (4.58.4)\n",
            "Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (1.4.8)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.11/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (24.2)\n",
            "Requirement already satisfied: pillow>=8 in /usr/local/lib/python3.11/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (11.2.1)\n",
            "Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (3.2.3)\n",
            "Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.11/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (2.9.0.post0)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.11/dist-packages (from pandas>=1.2->seaborn) (2025.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas>=1.2->seaborn) (2025.2)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil>=2.7->matplotlib!=3.6.1,>=3.4->seaborn) (1.17.0)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.decomposition import TruncatedSVD\n",
        "from sklearn.metrics.pairwise import cosine_similarity\n",
        "from IPython import get_ipython\n",
        "from IPython.display import display\n",
        "from collections import Counter\n",
        "from tabulate import tabulate"
      ],
      "metadata": {
        "id": "FnTPA6tl2Ckq"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **3. Memuat Dataset**"
      ],
      "metadata": {
        "id": "f3YIEnAFKrKL"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Pada tahap ini, Anda perlu memuat dataset ke dalam notebook. Jika dataset dalam format CSV, Anda bisa menggunakan pustaka pandas untuk membacanya. Pastikan untuk mengecek beberapa baris awal dataset untuk memahami strukturnya dan memastikan data telah dimuat dengan benar.\n",
        "\n",
        "Jika dataset berada di Google Drive, pastikan Anda menghubungkan Google Drive ke Colab terlebih dahulu. Setelah dataset berhasil dimuat, langkah berikutnya adalah memeriksa kesesuaian data dan siap untuk dianalisis lebih lanjut.\n",
        "\n",
        "Jika dataset berupa unstructured data, silakan sesuaikan dengan format seperti kelas Machine Learning Pengembangan atau Machine Learning Terapan"
      ],
      "metadata": {
        "id": "Ey3ItwTen_7E"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Load dataset\n",
        "movies = pd.read_csv('movies.csv')\n",
        "ratings = pd.read_csv('ratings.csv')\n",
        "tags = pd.read_csv('tags.csv')"
      ],
      "metadata": {
        "id": "GHCGNTyrM5fS"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **4. Exploratory Data Analysis (EDA)**\n",
        "\n",
        "Pada tahap ini, Anda akan melakukan **Exploratory Data Analysis (EDA)** untuk memahami karakteristik dataset.\n",
        "\n",
        "Tujuan dari EDA adalah untuk memperoleh wawasan awal yang mendalam mengenai data dan menentukan langkah selanjutnya dalam analisis atau pemodelan."
      ],
      "metadata": {
        "id": "bgZkbJLpK9UR"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Exploratory Data Analysis Dataset movies**"
      ],
      "metadata": {
        "id": "SHGm5A_X2qIX"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Dataset Info:\")\n",
        "print(movies.info())\n",
        "print(\"\\nDataset Head:\")\n",
        "print(movies.head())\n",
        "print(\"\\nDescriptive Statistics:\")\n",
        "print(movies.describe())"
      ],
      "metadata": {
        "id": "dKeejtvxM6X1",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "479e7c05-bfaa-40dd-dfa4-3c48f1fa2b12"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Dataset Info:\n",
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 62423 entries, 0 to 62422\n",
            "Data columns (total 3 columns):\n",
            " #   Column   Non-Null Count  Dtype \n",
            "---  ------   --------------  ----- \n",
            " 0   movieId  62423 non-null  int64 \n",
            " 1   title    62423 non-null  object\n",
            " 2   genres   62423 non-null  object\n",
            "dtypes: int64(1), object(2)\n",
            "memory usage: 1.4+ MB\n",
            "None\n",
            "\n",
            "Dataset Head:\n",
            "   movieId                               title  \\\n",
            "0        1                    Toy Story (1995)   \n",
            "1        2                      Jumanji (1995)   \n",
            "2        3             Grumpier Old Men (1995)   \n",
            "3        4            Waiting to Exhale (1995)   \n",
            "4        5  Father of the Bride Part II (1995)   \n",
            "\n",
            "                                        genres  \n",
            "0  Adventure|Animation|Children|Comedy|Fantasy  \n",
            "1                   Adventure|Children|Fantasy  \n",
            "2                               Comedy|Romance  \n",
            "3                         Comedy|Drama|Romance  \n",
            "4                                       Comedy  \n",
            "\n",
            "Descriptive Statistics:\n",
            "             movieId\n",
            "count   62423.000000\n",
            "mean   122220.387646\n",
            "std     63264.744844\n",
            "min         1.000000\n",
            "25%     82146.500000\n",
            "50%    138022.000000\n",
            "75%    173222.000000\n",
            "max    209171.000000\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Cek jumlah baris duplikat di Dataset movies\n",
        "jumlah_duplikat = movies.duplicated().sum()\n",
        "print(f\"Jumlah baris duplikat Dataset movies: {jumlah_duplikat}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "uaQ3I3FX2yPM",
        "outputId": "18cace79-05f9-4b02-e47a-2613bcfdda8d"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Jumlah baris duplikat Dataset movies: 0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Menampilkan jumlah missing value per kolom di Dataset movies\n",
        "missing_values = movies.isnull().sum()\n",
        "print(\"Jumlah Missing Value per kolom Dataset movies:\\n\")\n",
        "print(missing_values)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "B9-VPc_k22ov",
        "outputId": "4bc989d8-fb10-4c12-cd7d-55bebbd0f5c3"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Jumlah Missing Value per kolom Dataset movies:\n",
            "\n",
            "movieId    0\n",
            "title      0\n",
            "genres     0\n",
            "dtype: int64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Exploratory Data Analysis Dataset tags**"
      ],
      "metadata": {
        "id": "3G9HQCsw292w"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Dataset Info:\")\n",
        "print(tags.info())\n",
        "print(\"\\nDataset Head:\")\n",
        "print(tags.head())\n",
        "print(\"\\nDescriptive Statistics:\")\n",
        "print(tags.describe())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cvySLJfb2_FC",
        "outputId": "6c0caf2a-ce63-4819-ea1c-661c3ca1ff78"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Dataset Info:\n",
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 1093360 entries, 0 to 1093359\n",
            "Data columns (total 4 columns):\n",
            " #   Column     Non-Null Count    Dtype \n",
            "---  ------     --------------    ----- \n",
            " 0   userId     1093360 non-null  int64 \n",
            " 1   movieId    1093360 non-null  int64 \n",
            " 2   tag        1093344 non-null  object\n",
            " 3   timestamp  1093360 non-null  int64 \n",
            "dtypes: int64(3), object(1)\n",
            "memory usage: 33.4+ MB\n",
            "None\n",
            "\n",
            "Dataset Head:\n",
            "   userId  movieId               tag   timestamp\n",
            "0       3      260           classic  1439472355\n",
            "1       3      260            sci-fi  1439472256\n",
            "2       4     1732       dark comedy  1573943598\n",
            "3       4     1732    great dialogue  1573943604\n",
            "4       4     7569  so bad it's good  1573943455\n",
            "\n",
            "Descriptive Statistics:\n",
            "             userId       movieId     timestamp\n",
            "count  1.093360e+06  1.093360e+06  1.093360e+06\n",
            "mean   6.759022e+04  5.849276e+04  1.430115e+09\n",
            "std    5.152114e+04  5.968731e+04  1.177384e+08\n",
            "min    3.000000e+00  1.000000e+00  1.135429e+09\n",
            "25%    1.520400e+04  3.504000e+03  1.339262e+09\n",
            "50%    6.219900e+04  4.594000e+04  1.468929e+09\n",
            "75%    1.136420e+05  1.029030e+05  1.527402e+09\n",
            "max    1.625340e+05  2.090630e+05  1.574317e+09\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Cek jumlah baris duplikat di Dataset tags\n",
        "jumlah_duplikat = tags.duplicated().sum()\n",
        "print(f\"Jumlah baris duplikat Dataset tags: {jumlah_duplikat}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "BFKUeZzO3P2C",
        "outputId": "a76adf5f-5aee-462d-dff7-c177a8b6a1f6"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Jumlah baris duplikat Dataset tags: 0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Menampilkan jumlah missing value per kolom di Dataset tags\n",
        "missing_values = tags.isnull().sum()\n",
        "print(\"Jumlah Missing Value per kolom Dataset tags:\\n\")\n",
        "print(missing_values)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-gapxf843TU_",
        "outputId": "aff0024c-86f7-46bf-afdc-6b9bdeed8cd3"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Jumlah Missing Value per kolom Dataset tags:\n",
            "\n",
            "userId        0\n",
            "movieId       0\n",
            "tag          16\n",
            "timestamp     0\n",
            "dtype: int64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Exploratory Data Analysis Dataset ratings**"
      ],
      "metadata": {
        "id": "rMVetW_W3eU_"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Dataset Info:\")\n",
        "print(ratings.info())\n",
        "print(\"\\nDataset Head:\")\n",
        "print(ratings.head())\n",
        "print(\"\\nDescriptive Statistics:\")\n",
        "print(ratings.describe())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "hiuPQGYJ3g9s",
        "outputId": "55819b84-f24c-4662-f3e3-d9bb9c89ecab"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Dataset Info:\n",
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 25000095 entries, 0 to 25000094\n",
            "Data columns (total 4 columns):\n",
            " #   Column     Dtype  \n",
            "---  ------     -----  \n",
            " 0   userId     int64  \n",
            " 1   movieId    int64  \n",
            " 2   rating     float64\n",
            " 3   timestamp  int64  \n",
            "dtypes: float64(1), int64(3)\n",
            "memory usage: 762.9 MB\n",
            "None\n",
            "\n",
            "Dataset Head:\n",
            "   userId  movieId  rating   timestamp\n",
            "0       1      296     5.0  1147880044\n",
            "1       1      306     3.5  1147868817\n",
            "2       1      307     5.0  1147868828\n",
            "3       1      665     5.0  1147878820\n",
            "4       1      899     3.5  1147868510\n",
            "\n",
            "Descriptive Statistics:\n",
            "             userId       movieId        rating     timestamp\n",
            "count  2.500010e+07  2.500010e+07  2.500010e+07  2.500010e+07\n",
            "mean   8.118928e+04  2.138798e+04  3.533854e+00  1.215601e+09\n",
            "std    4.679172e+04  3.919886e+04  1.060744e+00  2.268758e+08\n",
            "min    1.000000e+00  1.000000e+00  5.000000e-01  7.896520e+08\n",
            "25%    4.051000e+04  1.196000e+03  3.000000e+00  1.011747e+09\n",
            "50%    8.091400e+04  2.947000e+03  3.500000e+00  1.198868e+09\n",
            "75%    1.215570e+05  8.623000e+03  4.000000e+00  1.447205e+09\n",
            "max    1.625410e+05  2.091710e+05  5.000000e+00  1.574328e+09\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Cek jumlah baris duplikat di Dataset ratings\n",
        "jumlah_duplikat = ratings.duplicated().sum()\n",
        "print(f\"Jumlah baris duplikat di Dataset ratings: {jumlah_duplikat}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "QK9nKsVr3gta",
        "outputId": "8fa43a12-0ab8-40d0-bd5a-f0f774db0725"
      },
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Jumlah baris duplikat di Dataset ratings: 0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Menampilkan jumlah missing value per kolom di Dataset ratings\n",
        "missing_values = ratings.isnull().sum()\n",
        "print(\"Jumlah Missing Value per Kolom Dataset ratings:\\n\")\n",
        "print(missing_values)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "im1-1dS33gLc",
        "outputId": "97b6cbc0-97c9-4548-fd18-2444aedb783c"
      },
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Jumlah Missing Value per Kolom Dataset ratings:\n",
            "\n",
            "userId       0\n",
            "movieId      0\n",
            "rating       0\n",
            "timestamp    0\n",
            "dtype: int64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def show_outliers(df, column):\n",
        "    # Periksa apakah kolom ada di DataFrame\n",
        "    if column not in df.columns:\n",
        "        print(f\"Error: Kolom '{column}' tidak ditemukan di DataFrame.\")\n",
        "        return pd.DataFrame() # Kembalikan DataFrame kosong jika kolom tidak ada\n",
        "\n",
        "    Q1 = df[column].quantile(0.25)\n",
        "    Q3 = df[column].quantile(0.75)\n",
        "    IQR = Q3 - Q1\n",
        "    lower_bound = Q1 - 1.5 * IQR\n",
        "    upper_bound = Q3 + 1.5 * IQR\n",
        "\n",
        "    # Filter outlier dari DataFrame yang diberikan (df), bukan 'movies'\n",
        "    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]\n",
        "    print(f\"Jumlah outlier pada kolom {column}: {len(outliers)}\")\n",
        "    return outliers\n",
        "\n",
        "# Contoh penggunaan untuk mencari outlier di kolom 'rating' pada DataFrame 'ratings'\n",
        "outlier_data_ratings = show_outliers(ratings, 'rating')\n",
        "display(outlier_data_ratings.head())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 224
        },
        "id": "cl3QHotS3wOJ",
        "outputId": "f6fb6c26-afe2-4bba-a9ec-887695a1acfb"
      },
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Jumlah outlier pada kolom rating: 1169883\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "    userId  movieId  rating   timestamp\n",
              "31       1     5269     0.5  1147879571\n",
              "60       1     8685     1.0  1147878023\n",
              "71       2       62     0.5  1141417130\n",
              "77       2      261     0.5  1141417855\n",
              "78       2      266     1.0  1141415926"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-cea97e8f-fe8d-4c8d-859e-4153e327a78f\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>userId</th>\n",
              "      <th>movieId</th>\n",
              "      <th>rating</th>\n",
              "      <th>timestamp</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>31</th>\n",
              "      <td>1</td>\n",
              "      <td>5269</td>\n",
              "      <td>0.5</td>\n",
              "      <td>1147879571</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>60</th>\n",
              "      <td>1</td>\n",
              "      <td>8685</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1147878023</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>71</th>\n",
              "      <td>2</td>\n",
              "      <td>62</td>\n",
              "      <td>0.5</td>\n",
              "      <td>1141417130</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>77</th>\n",
              "      <td>2</td>\n",
              "      <td>261</td>\n",
              "      <td>0.5</td>\n",
              "      <td>1141417855</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>78</th>\n",
              "      <td>2</td>\n",
              "      <td>266</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1141415926</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-cea97e8f-fe8d-4c8d-859e-4153e327a78f')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-cea97e8f-fe8d-4c8d-859e-4153e327a78f button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-cea97e8f-fe8d-4c8d-859e-4153e327a78f');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "    <div id=\"df-4714ff2a-eada-4359-8706-f5cb1c95a35e\">\n",
              "      <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-4714ff2a-eada-4359-8706-f5cb1c95a35e')\"\n",
              "                title=\"Suggest charts\"\n",
              "                style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "      </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "      <script>\n",
              "        async function quickchart(key) {\n",
              "          const quickchartButtonEl =\n",
              "            document.querySelector('#' + key + ' button');\n",
              "          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "          quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "          try {\n",
              "            const charts = await google.colab.kernel.invokeFunction(\n",
              "                'suggestCharts', [key], {});\n",
              "          } catch (error) {\n",
              "            console.error('Error during call to suggestCharts:', error);\n",
              "          }\n",
              "          quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "          quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "        }\n",
              "        (() => {\n",
              "          let quickchartButtonEl =\n",
              "            document.querySelector('#df-4714ff2a-eada-4359-8706-f5cb1c95a35e button');\n",
              "          quickchartButtonEl.style.display =\n",
              "            google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "        })();\n",
              "      </script>\n",
              "    </div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "summary": "{\n  \"name\": \"display(outlier_data_ratings\",\n  \"rows\": 5,\n  \"fields\": [\n    {\n      \"column\": \"userId\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 1,\n        \"max\": 2,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          2,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"movieId\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 3906,\n        \"min\": 62,\n        \"max\": 8685,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          8685,\n          266\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"rating\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0.27386127875258304,\n        \"min\": 0.5,\n        \"max\": 1.0,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1.0,\n          0.5\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"timestamp\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 3539288,\n        \"min\": 1141415926,\n        \"max\": 1147879571,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          1147878023,\n          1141415926\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **5. Data Preprocessing**"
      ],
      "metadata": {
        "id": "cpgHfgnSK3ip"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Pada tahap ini, data preprocessing adalah langkah penting untuk memastikan kualitas data sebelum digunakan dalam model machine learning.\n",
        "\n",
        "Jika Anda menggunakan data teks, data mentah sering kali mengandung nilai kosong, duplikasi, atau rentang nilai yang tidak konsisten, yang dapat memengaruhi kinerja model. Oleh karena itu, proses ini bertujuan untuk membersihkan dan mempersiapkan data agar analisis berjalan optimal.\n",
        "\n",
        "Berikut adalah tahapan-tahapan yang bisa dilakukan, tetapi **tidak terbatas** pada:\n",
        "1. Menghapus atau Menangani Data Kosong (Missing Values)\n",
        "2. Menghapus Data Duplikat\n",
        "3. Normalisasi atau Standarisasi Fitur\n",
        "4. Deteksi dan Penanganan Outlier\n",
        "5. Encoding Data Kategorikal\n",
        "6. Binning (Pengelompokan Data)\n",
        "\n",
        "Cukup sesuaikan dengan karakteristik data yang kamu gunakan yah. Khususnya ketika kami menggunakan data tidak terstruktur."
      ],
      "metadata": {
        "id": "COf8KUPXLg5r"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Langkah 1: Persiapan\n",
        "def load_and_prepare_data(movies_path='movies.csv', ratings_path='ratings.csv', tags_path='tags.csv'):\n",
        "    import pandas as pd # Import pandas here\n",
        "    # Load dataset\n",
        "    movies = pd.read_csv('movies.csv')\n",
        "    ratings = pd.read_csv('ratings.csv')\n",
        "    tags = pd.read_csv('tags.csv')\n",
        "\n",
        "    # Menghapus duplikasi pada tags\n",
        "    tags_cleaned = tags.drop_duplicates()\n",
        "\n",
        "    # Mengatasi missing value pada tags\n",
        "    tags['tag'] = tags['tag'].fillna('no_tag')\n",
        "\n",
        "    # Menghapus baris dengan tag kosong/null\n",
        "    tags_cleaned = tags_cleaned.dropna(subset=['tag'])\n",
        "\n",
        "    # Menangani genre dengan nilai '(no genres listed)'\n",
        "    movies_cleaned = movies[movies['genres'] != '(no genres listed)'].copy()\n",
        "\n",
        "    # Menyatukan genre ke dalam format token terpisah\n",
        "    # Menambahkan kolom 'genre_tokens' yang isinya list genre\n",
        "    movies_cleaned['genre_tokens'] = movies_cleaned['genres'].apply(lambda x: x.split('|'))\n",
        "\n",
        "    # Konversi teks tag menjadi huruf kecil\n",
        "    tags_cleaned['tag'] = tags_cleaned['tag'].str.lower()\n",
        "\n",
        "    # Filter film populer (minimal 100 rating untuk mengurangi ukuran dataset)\n",
        "    rating_counts = ratings['movieId'].value_counts()\n",
        "    popular_movies = rating_counts[rating_counts > 100].index\n",
        "    movies = movies[movies['movieId'].isin(popular_movies)].copy() # Use .copy() to avoid SettingWithCopyWarning\n",
        "\n",
        "    # Gabungkan genre dan tag\n",
        "    movies['genres'] = movies['genres'].str.replace('|', ' ')\n",
        "\n",
        "    # Ensure tags data is filtered to popular movies before merging\n",
        "    # Convert 'tag' column to string type and fill NaN with empty strings\n",
        "    tags_filtered = tags[tags['movieId'].isin(popular_movies)].copy() # Filter tags and use .copy()\n",
        "    tags_filtered['tag'] = tags_filtered['tag'].astype(str).fillna('')\n",
        "\n",
        "    movie_tags = tags_filtered.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()\n",
        "    movies = movies.merge(movie_tags, on='movieId', how='left')\n",
        "    movies['tags'] = movies['tag'].fillna('')\n",
        "    movies = movies.drop('tag', axis=1) # Drop the temporary 'tag' column after merging\n",
        "\n",
        "    # Combine genres and tags into a single content column for TF-IDF\n",
        "    # Ensure 'genres' and 'tags' columns exist before combining\n",
        "    movies['content'] = movies['genres'] + ' ' + movies['tags']\n",
        "\n",
        "    return movies, ratings, tags # Return all three DataFrames"
      ],
      "metadata": {
        "id": "Og8pGV0-iDLz"
      },
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Modify build_feature_matrix to return the fitted SVD object as well\n",
        "def build_feature_matrix(movies, max_features=1000, n_components=100):\n",
        "    # Ekstraksi fitur dengan TF-IDF\n",
        "    tfidf = TfidfVectorizer(stop_words='english', max_features=max_features)\n",
        "    tfidf_matrix = tfidf.fit_transform(movies['content'])\n",
        "\n",
        "    # Reduksi dimensi dengan TruncatedSVD\n",
        "    svd = TruncatedSVD(n_components=n_components)\n",
        "    tfidf_matrix_reduced = svd.fit_transform(tfidf_matrix)\n",
        "\n",
        "    # Return the reduced matrix, the tfidf vectorizer, and the fitted svd object\n",
        "    return tfidf_matrix_reduced, tfidf, svd\n"
      ],
      "metadata": {
        "id": "1siA2m2x5bju"
      },
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Langkah 3: Hitung Cosine Similarity\n",
        "def compute_cosine_similarity(tfidf_matrix_reduced):\n",
        "    cosine_sim = cosine_similarity(tfidf_matrix_reduced, tfidf_matrix_reduced)\n",
        "    return cosine_sim\n"
      ],
      "metadata": {
        "id": "sanB1R1J5gca"
      },
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Rekomendasi untuk Pengguna Existing (Pertanyaan 1)\n",
        "def get_existing_user_recommendations(user_id, movies, ratings, tfidf_matrix_reduced, cosine_sim, top_n=10):\n",
        "    # Ambil film yang disukai pengguna (rating >= 4)\n",
        "    user_ratings = ratings[ratings['userId'] == user_id]\n",
        "    liked_movies = user_ratings[user_ratings['rating'] >= 4]['movieId']\n",
        "    user_indices = movies[movies['movieId'].isin(liked_movies)].index\n",
        "\n",
        "    if len(user_indices) == 0:\n",
        "        return \"Pengguna belum memiliki film yang disukai.\"\n",
        "\n",
        "    # Buat profil pengguna\n",
        "    user_profile = tfidf_matrix_reduced[user_indices].mean(axis=0).reshape(1, -1)\n",
        "    sim_scores = cosine_similarity(user_profile, tfidf_matrix_reduced)[0]\n",
        "    sim_scores = list(enumerate(sim_scores))\n",
        "    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)\n",
        "\n",
        "    # Ambil film yang belum ditonton\n",
        "    movie_indices = [i[0] for i in sim_scores if i[0] not in user_indices][:top_n]\n",
        "    return movies[['title', 'genres']].iloc[movie_indices]\n"
      ],
      "metadata": {
        "id": "0y8OEOcd5k0X"
      },
      "execution_count": 17,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Add the fitted svd object as an argument\n",
        "def get_new_user_recommendations(preferred_genres, movies, tfidf, tfidf_matrix_reduced, fitted_svd, cosine_sim, top_n=10):\n",
        "    # Ubah preferensi genre pengguna baru menjadi vektor TF-IDF\n",
        "    user_input = ' '.join(preferred_genres)\n",
        "    user_tfidf = tfidf.transform([user_input])\n",
        "\n",
        "    # Use the pre-fitted SVD object to transform the user input\n",
        "    user_tfidf_reduced = fitted_svd.transform(user_tfidf)\n",
        "\n",
        "    # Hitung kemiripan dengan semua film\n",
        "    sim_scores = cosine_similarity(user_tfidf_reduced, tfidf_matrix_reduced)[0]\n",
        "    sim_scores = list(enumerate(sim_scores))\n",
        "    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)\n",
        "    movie_indices = [i[0] for i in sim_scores[:top_n]]\n",
        "    return movies[['title', 'genres']].iloc[movie_indices]"
      ],
      "metadata": {
        "id": "xrQuG3ou5ruR"
      },
      "execution_count": 18,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Langkah 6: Evaluasi Rekomendasi (Precision@10)\n",
        "def evaluate_recommendations(user_id, movies, ratings, recommended_indices, top_n=10):\n",
        "    user_ratings = ratings[ratings['userId'] == user_id]\n",
        "    liked_movies = user_ratings[user_ratings['rating'] >= 4]['movieId']\n",
        "    recommended_movies = movies.iloc[recommended_indices]['movieId'].values\n",
        "\n",
        "    # Convert both sets of movie IDs to sets for efficient intersection\n",
        "    liked_movie_ids = set(liked_movies)\n",
        "    recommended_movie_ids = set(recommended_movies)\n",
        "\n",
        "    # Calculate the number of hits\n",
        "    hits = len(recommended_movie_ids.intersection(liked_movie_ids))\n",
        "\n",
        "    # Calculate precision\n",
        "    precision = hits / top_n if top_n > 0 else 0\n",
        "    return precision\n"
      ],
      "metadata": {
        "id": "JXLDZtuf5xui"
      },
      "execution_count": 19,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Main Program\n",
        "if __name__ == \"__main__\":\n",
        "    # Load and prepare data\n",
        "    movies, ratings, tags = load_and_prepare_data()\n",
        "\n",
        "    if 'content' not in movies.columns:\n",
        "        print(\"❌ Error: Kolom 'content' belum dibuat dalam DataFrame movies.\")\n",
        "    else:\n",
        "        # Ekstraksi fitur dan reduksi dimensi\n",
        "        tfidf_matrix_reduced, tfidf, fitted_svd = build_feature_matrix(movies)\n",
        "\n",
        "        # Hitung cosine similarity\n",
        "        cosine_sim = compute_cosine_similarity(tfidf_matrix_reduced)\n",
        "\n",
        "        ### --- REKOMENDASI UNTUK PENGGUNA EXISTING --- ###\n",
        "        user_id = 1\n",
        "        print(f\"\\n🎯 Top-N Rekomendasi untuk Pengguna Existing (User ID: {user_id}):\")\n",
        "        if user_id in ratings['userId'].unique():\n",
        "            existing_user_recommendations = get_existing_user_recommendations(\n",
        "                user_id, movies, ratings, tfidf_matrix_reduced, cosine_sim\n",
        "            )\n",
        "\n",
        "            if isinstance(existing_user_recommendations, pd.DataFrame):\n",
        "                # Ambil Top-N\n",
        "                top_n = 10\n",
        "                top_n_recs = existing_user_recommendations.head(top_n)\n",
        "\n",
        "                # Tampilkan sebagai tabel\n",
        "                print(tabulate(top_n_recs, headers='keys', tablefmt='github', showindex=False))\n",
        "\n",
        "                # Evaluasi Precision@10\n",
        "                rec_indices = top_n_recs.index\n",
        "                precision = evaluate_recommendations(\n",
        "                    user_id=user_id,\n",
        "                    movies=movies,\n",
        "                    ratings=ratings,\n",
        "                    recommended_indices=rec_indices\n",
        "                )\n",
        "                print(f\"\\n📊 Precision@{top_n} untuk User ID {user_id}: **{precision:.2f}**\")\n",
        "            else:\n",
        "                print(existing_user_recommendations)\n",
        "        else:\n",
        "            print(\"⚠️ User ID tidak ditemukan dalam data ratings.\")\n",
        "\n",
        "        ### --- REKOMENDASI UNTUK PENGGUNA BARU --- ###\n",
        "        preferred_genres = ['Action', 'Sci-Fi', 'Adventure']\n",
        "        print(f\"\\n✨ Top-N Rekomendasi untuk Pengguna Baru (Preferensi: {', '.join(preferred_genres)}):\")\n",
        "        new_user_recommendations = get_new_user_recommendations(\n",
        "            preferred_genres,\n",
        "            movies,\n",
        "            tfidf,\n",
        "            tfidf_matrix_reduced,\n",
        "            fitted_svd,\n",
        "            cosine_sim\n",
        "        )\n",
        "\n",
        "        if isinstance(new_user_recommendations, pd.DataFrame):\n",
        "            # Ambil Top-N\n",
        "            top_n = 10\n",
        "            top_n_recs_new_user = new_user_recommendations.head(top_n)\n",
        "\n",
        "            # Tampilkan sebagai tabel\n",
        "            print(tabulate(top_n_recs_new_user, headers='keys', tablefmt='github', showindex=False))\n",
        "        else:\n",
        "            print(new_user_recommendations)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ZOaQl_0l57B_",
        "outputId": "fcd0782a-5899-49c5-d835-116287e73dab"
      },
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "🎯 Top-N Rekomendasi untuk Pengguna Existing (User ID: 1):\n",
            "| title                                                              | genres                |\n",
            "|--------------------------------------------------------------------|-----------------------|\n",
            "| Chungking Express (Chung Hing sam lam) (1994)                      | Drama Mystery Romance |\n",
            "| Double Life of Veronique, The (Double Vie de Véronique, La) (1991) | Drama Fantasy Romance |\n",
            "| Europa (Zentropa) (1991)                                           | Drama Thriller        |\n",
            "| Harder They Come, The (1973)                                       | Action Crime Drama    |\n",
            "| Black Narcissus (1947)                                             | Drama                 |\n",
            "| Winter Light (Nattvardsgästerna) (1963)                            | Drama                 |\n",
            "| Andrei Rublev (Andrey Rublyov) (1969)                              | Drama War             |\n",
            "| Days of Heaven (1978)                                              | Drama                 |\n",
            "| Persona (1966)                                                     | Drama                 |\n",
            "| Black Orpheus (Orfeu Negro) (1959)                                 | Drama Romance         |\n",
            "\n",
            "📊 Precision@10 untuk User ID 1: **0.00**\n",
            "\n",
            "✨ Top-N Rekomendasi untuk Pengguna Baru (Preferensi: Action, Sci-Fi, Adventure):\n",
            "| title                                                 | genres                                  |\n",
            "|-------------------------------------------------------|-----------------------------------------|\n",
            "| Adventures of Pluto Nash, The (2002)                  | Action Adventure Comedy Sci-Fi          |\n",
            "| Princess Blade, The (Shura Yukihime) (2001)           | Action Sci-Fi                           |\n",
            "| Baby... Secret of the Lost Legend (1985)              | Adventure Sci-Fi                        |\n",
            "| Bumblebee (2018)                                      | Action Adventure Sci-Fi                 |\n",
            "| Stargate SG-1 Children of the Gods - Final Cut (2009) | Adventure Sci-Fi Thriller               |\n",
            "| Jurassic World (2015)                                 | Action Adventure Drama Sci-Fi Thriller  |\n",
            "| Push (2009)                                           | Sci-Fi Thriller                         |\n",
            "| Clockstoppers (2002)                                  | Action Adventure Sci-Fi Thriller        |\n",
            "| Jurassic Park (1993)                                  | Action Adventure Sci-Fi Thriller        |\n",
            "| Honey, We Shrunk Ourselves (1997)                     | Action Adventure Children Comedy Sci-Fi |\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "%%writefile automate_Heriyanto.py\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.preprocessing import OneHotEncoder, MinMaxScaler\n",
        "from sklearn.impute import SimpleImputer\n",
        "\n",
        "def handle_missing_values(df, categorical_cols, numerical_cols):\n",
        "    \"\"\"\n",
        "    Mengisi missing values untuk kolom numerik dan kategorikal\n",
        "    \"\"\"\n",
        "    df = df.copy()\n",
        "\n",
        "    # Imputasi numerik\n",
        "    num_imputer = SimpleImputer(strategy='mean')\n",
        "    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])\n",
        "\n",
        "    # Imputasi kategorikal\n",
        "    cat_imputer = SimpleImputer(strategy='most_frequent')\n",
        "    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])\n",
        "\n",
        "    return df\n",
        "\n",
        "def encode_categorical_features(df, categorical_cols):\n",
        "    \"\"\"\n",
        "    One-hot encoding untuk kolom kategorikal\n",
        "    \"\"\"\n",
        "    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')\n",
        "    encoded_array = encoder.fit_transform(df[categorical_cols])\n",
        "    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_cols))\n",
        "    return encoded_df, encoder\n",
        "\n",
        "def normalize_numeric_features(df, numerical_cols):\n",
        "    \"\"\"\n",
        "    Normalisasi fitur numerik dengan MinMaxScaler\n",
        "    \"\"\"\n",
        "    scaler = MinMaxScaler()\n",
        "    scaled_array = scaler.fit_transform(df[numerical_cols])\n",
        "    scaled_df = pd.DataFrame(scaled_array, columns=[f\"{col}_scaled\" for col in numerical_cols])\n",
        "    return scaled_df, scaler\n",
        "\n",
        "def preprocess_movie_data(df, categorical_cols=['genre'], numerical_cols=['duration', 'rating', 'budget']):\n",
        "    \"\"\"\n",
        "    Pipeline preprocessing utama, mengembalikan dataframe siap latih\n",
        "    \"\"\"\n",
        "    df = df.copy()\n",
        "\n",
        "    # Step 1: Missing value handling\n",
        "    df = handle_missing_values(df, categorical_cols, numerical_cols)\n",
        "\n",
        "    # Step 2: Encoding kategorikal\n",
        "    encoded_df, _ = encode_categorical_features(df, categorical_cols)\n",
        "\n",
        "    # Step 3: Normalisasi numerik\n",
        "    scaled_df, _ = normalize_numeric_features(df, numerical_cols)\n",
        "\n",
        "    # Gabungkan hasil akhir\n",
        "    df_final = pd.concat([\n",
        "        df.drop(columns=categorical_cols + numerical_cols).reset_index(drop=True),\n",
        "        encoded_df.reset_index(drop=True),\n",
        "        scaled_df.reset_index(drop=True)\n",
        "    ], axis=1)\n",
        "\n",
        "    return df_final\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cBd4aM1Cu4OB",
        "outputId": "1af06855-9e11-4fb1-d0fe-07e59ac96742"
      },
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Writing automate_Heriyanto.py\n"
          ]
        }
      ]
    }
  ]
}