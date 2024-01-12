import os
from newspaper import Article
from bs4 import BeautifulSoup
from konlpy.tag import Okt
from gensim.models import word2vec
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# Function to create a directory if it doesn't exist
def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

# Function to save a list of tuples to a text file
def save_result_to_file(result, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write("---------------------------- Result --------------------------------\n\n")
        for item in result:
            file.write(f"{item[0]} : {item[1]}\n")

# Function to show a bar graph
def show_graph(bargraph, result_dir):
    font_path = "/home/sm32289/GDSC_Newyear/Emotion/txtfile/NanumGothic.ttf"
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)

    xtick = [item[0] for item in bargraph]  # 단어
    ytick = [item[1] for item in bargraph]  # 유사도

    plt.figure()

    mycolors = ['#06c2ac', '#c79fef', '#ff796c', '#aaff32', '#0485d1', '#d648d7', '#a5a502', '#d8dcd6', '#5ca904', '#fffe7a']

    plt.bar(xtick, ytick, color=mycolors)

    # Save the result to a file
    file_number = len(os.listdir(result_dir)) + 1
    result_file_name = f"result{file_number}.txt"
    result_file_path = os.path.join(result_dir, result_file_name)
    save_result_to_file(bargraph, result_file_path)

    plt.show()

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Specify directories using the relative paths
news_articles_dir = os.path.join(script_dir, 'News_articles')
combine_data_dir = os.path.join(script_dir, 'combine_data')
word2vec_model_dir = os.path.join(script_dir, 'word2vecmodel')
train_setting_dir = os.path.join(script_dir, 'train_setting')
result_dir = os.path.join(script_dir, 'result')

# Create directories if they don't exist
create_directory(news_articles_dir)
create_directory(combine_data_dir)
create_directory(word2vec_model_dir)
create_directory(train_setting_dir)
create_directory(result_dir)

# News articles URLs
URL = [
    "https://www.1conomynews.co.kr/news/articleView.html?idxno=25772",
    "https://www.yna.co.kr/view/AKR20240112045600063",
    "https://mdtoday.co.kr/news/view/1065578601572123",
    "https://www.cctoday.co.kr/news/articleView.html?idxno=2181067"
]

# Download and save news articles
for i, url in enumerate(URL):
    article = Article(url, language='ko')
    article.download()
    article.parse()

    news_title = article.title
    news_context = article.text

    # Generate a file name with a unique number
    file_number = i + 1  # Start numbering from 1
    file_name = os.path.join(news_articles_dir, f"news_article{file_number}.txt")

    with open(file_name, "w", encoding="utf-8") as file:
        file.write("Title: " + news_title + "\n\n")
        file.write(news_context)

    print("News article has been saved to", file_name)

# Combine news articles into a single preprocessed file
train_article = len(os.listdir(news_articles_dir))
result = []
for filename in os.listdir(news_articles_dir):
    if filename.endswith('.txt'):
        filepath = os.path.join(news_articles_dir, filename)
        with open(filepath, 'rt', encoding='utf-8') as myfile:
            soup = BeautifulSoup(myfile, 'html.parser')
            mydata = soup.text

        okt = Okt()
        detalines = mydata.split('\n')

        for oneline in detalines:
            mypos = okt.pos(oneline, norm=True, stem=True)
            tmp = [word[0] for word in mypos if word[1] not in ['Josa', 'Eomi', 'Punctuation', 'Verb'] and len(word[0]) >= 2]
            result.append(' '.join(tmp).strip())

# Save preprocessed text to a file
result_file = os.path.join(combine_data_dir, f'word2vecTest.prepro{len(os.listdir(combine_data_dir)) + 1}.txt')
with open(result_file, 'wt', encoding='utf-8') as myfile:
    myfile.write('\n'.join(result) + '\n')

print(result_file + ' 파일 저장됨!')

# Train Word2Vec model
data = word2vec.LineSentence(source=result_file)
model = word2vec.Word2Vec(data, vector_size=200, window=10, hs=1, min_count=2, sg=1)

# Save the trained model
file_list = os.listdir(word2vec_model_dir)
file_count = len(file_list)
model_filename = f'word2vec.model{file_count + 1}'
model_save_dir = os.path.join(word2vec_model_dir, model_filename)
model.save(model_save_dir)
print(model_save_dir + ' 모델 저장됨!')

# Save train setting information
train_setting_file = os.path.join(train_setting_dir, f"news_article{len(os.listdir(train_setting_dir)) + 1}.txt")
