import json
import inflect
import pandas as pd
from datetime import datetime as dt, timedelta
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import unicodedata
from nltk.stem import WordNetLemmatizer
import re
import math
import plotly.express as px
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nltk import FreqDist, LancasterStemmer


print("Data preprocessing")

with open('final_data.json', 'r') as openfile:
    # Reading from json file
    data = json.load(openfile)

with open('final_meta_data.json', 'r') as openfile:
    # Reading from json file
    meta_data = json.load(openfile)

# Data processing

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text


# special_characters removal
def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation_and_splchars(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_word = remove_special_characters(new_word, True)
            new_words.append(new_word)
    return new_words


def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def remove_stopwords(words, stopword_list):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopword_list:
            new_words.append(word)
    return new_words


def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems


def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


def normalize(words, stopword_list):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation_and_splchars(words)
    words = remove_stopwords(words, stopword_list)
    return words


def lemmatize(words):
    lemmas = lemmatize_verbs(words)
    return lemmas


def normalize_and_lemmaize(input, stopword_list):
    sample = denoise_text(input)
    sample = remove_special_characters(sample)
    words = nltk.word_tokenize(sample)
    words = normalize(words, stopword_list)
    lemmas = lemmatize(words)
    return ' '.join(lemmas)


def get_meta_data(asin):
    drop = ['image', 'tech1', 'description', 'fit', 'image', 'tech2', 'brand', 'feature', 'details', 'date']
    temp = {}
    for item in meta_data:
        if item['asin'] == asin:
            for key in item.keys():
                if key not in drop:
                    temp[key] = item[key]
    return temp


def get_review_set(asin):
    subset = []
    for i in data:
        if i['asin'] == asin:
            subset.append(i)
    return pd.DataFrame(subset)


def get_data_for_asin(asin):
    asin_data = get_meta_data(asin)
    if asin_data == {} or len(asin_data['title']) == 0:
        return "Error : Product information not present"
    review_data = get_review_set(asin)
    if len(review_data) <= 20:
        return "Error: Only " + str(len(review_data)) + " reviews available"
    asin_data['review_set'] = review_data
    return asin_data


def remove_null(data_raw_):
    data_raw = data_raw_['review_set']
    to_be_dropped = []
    for i in range(0, len(data_raw)):
        row = data_raw.iloc[i]
        revTxt = str(row['reviewText']).lower()
        if len(revTxt) == 0 or revTxt == 'nan':
            to_be_dropped.append(i)
    data_raw_f = data_raw.drop(index=to_be_dropped)
    data_raw_['review_set'] = data_raw_f
    return data_raw_


def get_unique_reviews(null_cleaned):
    null_cleaned['date'] = null_cleaned['unixReviewTime'].transform(lambda x: dt.utcfromtimestamp(x))
    dup_count = null_cleaned.reviewerID.value_counts()
    invalid_revid = []
    for revId in dup_count.index:
        if dup_count[revId] > 1:
            subset = null_cleaned[null_cleaned['reviewerID'] == revId]
            max_dt = subset['date'].max()
            temp = subset
            temp['time_dif'] = max_dt - subset['date']
            if temp['time_dif'].min() < timedelta(days=30):
                invalid_revid.append(revId)
    for irevid in invalid_revid:
        if irevid in null_cleaned['reviewerID'].values :
            null_cleaned = null_cleaned.drop(list(null_cleaned[null_cleaned['reviewerID'] == irevid].index))
    return null_cleaned


def clean_data(data_raw):
    null_cleaned = remove_null(data_raw)
    unique_reviewset = get_unique_reviews(null_cleaned['review_set'])
    null_cleaned['review_set'] = unique_reviewset
    return null_cleaned


def combine(clean_revTxt, clean_revSum):
    rev = clean_revTxt
    for w in clean_revSum.split():
        if w not in clean_revTxt:
            rev = rev + w
    return rev


def wrangle_data(clean_):
    stopword_list = stopwords.words('english')
    stopword_list.remove('no')
    stopword_list.remove('not')
    stopword_list.append('amazon')
    stopword_list.append('product')
    stopword_list.append('five')
    stopword_list.append('star')
    stopword_list.append('use')
    stopword_list.append('go')
    clean_Title = normalize_and_lemmaize(clean_['title'], stopword_list)
    clean_['title'] = clean_Title
    for w in clean_Title.split():
        stopword_list.append(w)

    clean = clean_['review_set']
    # print(clean)
    helpful = []
    review = []
    for r in range(0, len(clean)):
        row = clean.iloc[r]
        try:
            vote = int(row.vote)
        except:
            vote = 0
        if vote <= 2:
            helpful.append(True)
        else:
            helpful.append(False)

        clean_revTxt = normalize_and_lemmaize(str(row.reviewText), stopword_list)
        clean_revSum = normalize_and_lemmaize(str(row.summary), stopword_list)
        review.append(clean_revSum + " " + clean_revTxt)

    clean['helpful'] = helpful
    clean['review'] = review
    clean_['review_set'] = clean
    return clean_


def summarise_data(all_reviews_, asin):
    all_reviews_good_bad = {}
    all_reviews_good_bad['good'] = all_reviews_[all_reviews_['overall'] > 3]
    all_reviews_good_bad['bad'] = all_reviews_[all_reviews_['overall'] <= 3]
    final_comb = {}
    for typ in all_reviews_good_bad.keys():
        final = {}
        reviews = []
        review_len = []
        imgCount = []
        all_reviews = all_reviews_good_bad[typ]
        final['helpful'] = list(all_reviews['helpful'])
        final['time_period'] = list(all_reviews['date'])
        final['ratings'] = list(all_reviews['overall'])
        imgCount = []
        review_len = []
        for r in range(0, len(all_reviews)):
            row = all_reviews.iloc[r]
            img_c = row.image
            if str(img_c).lower() == 'nan':
                img_c = 0
            else:
                img_c = len(img_c)
            imgCount.append(img_c)
            review_len.append(len(row.review.split()))
            reviews.append(row.review.split())
        final['image_count'] = imgCount
        final['review_word_count'] = review_len
        final['reviews'] = reviews
        final_comb[typ] = final
    return final_comb


def Bag_Of_Words(ListWords):
    all_words = []
    for m in ListWords:
        for w in m:
            all_words.append(w.lower())
    all_words1 = FreqDist(all_words)
    return all_words1

print("Pre-processing completed\n Analysis begins")

def getURL(img_name,storage,user):
    storage.child(img_name).put(img_name)
    url= storage.child(img_name).get_url(user['idToken'])
    return url

def get_recommendation_score(all_reviews):
    rel_score = []
    for r in range(0, len(all_reviews['review_set'])):
        row = all_reviews['review_set'].iloc[r]

        if str(row.image) == 'nan':
            img = 0
        else:
            img = len(row.image)

        score = row.helpful * (img + len(row.review.split()))
        rel_score.append(score)

    all_reviews['review_set']['reliabilityScore'] = rel_score
    all_reviews['review_set']['reliabilityIndex'] = (all_reviews['review_set']['reliabilityScore'] - min(
        all_reviews['review_set']['reliabilityScore'])) / (max(all_reviews['review_set']['reliabilityScore']) - min(
        all_reviews['review_set']['reliabilityScore']))

    recommendationScore = sum(
        all_reviews['review_set']['reliabilityIndex'] * all_reviews['review_set']['overall']) / len(
        all_reviews['review_set'])

    return recommendationScore


# Scatter plot function
def get_scatter_plot(all_reviews,storage,user,ts):
    date = all_reviews['review_set']['date']
    rating = all_reviews['review_set']['overall']
    size = all_reviews['review_set']['reliabilityIndex']

    fig = px.scatter(x=date, y=rating, color=size, size=size)
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Rating",
        title="Ratings over time, color and size based on reliabiltiyIndex",
        legend_title="")
    fig.write_html(ts + 'scatter_plot_html.html')
    url1 = getURL(ts + 'scatter_plot_html.html', storage, user)
    return url1


def get_word_cloud(review_set, img_name, storage, user, ts):
    if len(review_set) == 0:
        return "NULL"
    plot = plt
    bow = Bag_Of_Words(review_set)
    bow = dict(bow)
    row = {}
    word_df = pd.DataFrame()
    for a in bow:
        row['word'] = a
        row['count'] = bow[a]

        in_review = 0
        for review in review_set:
            # print(review)
            if a in review:
                # print(a,in_review)
                in_review = in_review + 1
        row['word_present_in'] = in_review
        word_df = word_df.append(row, ignore_index=True)

    word_df['frequency'] = word_df['count'] / sum(word_df['count'])
    word_df['idf'] = len(review_set) / word_df['word_present_in']
    word_df['idf'] = word_df['idf'].transform(lambda x: math.log(x))

    word_df['tf_idf_score'] = word_df['frequency'] * word_df['idf']
    tfidf = word_df[['word', 'tf_idf_score']]
    freq_Dist_tfidf = {}
    for r in range(0, len(tfidf)):
        row = tfidf.iloc[r]
        freq_Dist_tfidf[row['word']] = row['tf_idf_score']
    wordcloud = WordCloud(background_color='white', max_font_size=40).generate_from_frequencies(freq_Dist_tfidf)
    plot.imshow(wordcloud, interpolation='bilinear')
    plot.axis("off")
    plot.savefig(img_name + ".png")
    plt.close()
    url2 = getURL(img_name + ".png", storage, user)
    return url2


def get_pie_chart(values, labels, storage, user, ts):
    pie = plt.pie(values, explode=[0, 0.01], labels=labels, colors=['green', 'red'], autopct='%1.1f%%')
    pie = plt.title('Good Reviews Vs Bad Reviews')
    my_circle = plt.Circle((0, 0), 0.7, color='white')
    p = plt.gcf()
    pie = p.gca().add_artist(my_circle)
    p.savefig(ts + 'pie_chart_good_bad.png')
    plt.close()
    url3 = getURL(ts + "pie_chart_good_bad" + ".png", storage, user)
    return url3


def get_bar_plot(similar, storage, user, ts):
    fig = px.bar(similar, x='names', y='recommendation_score')
    fig.write_html(ts + 'bar_plot.html')
    url4 = getURL(ts + 'bar_plot.html', storage, user)
    return url4


def get_rank_from(text, cat_txt):
    number = ''
    text = text.replace(',', '')
    for ch in text:
        if ch.isnumeric():
            number = number + ch
    meta_data_df = pd.DataFrame(meta_data)
    total = len(meta_data_df[meta_data_df.main_cat == cat_txt])
    return [int(number), total]


def get_sales_pot_chart(sales_potential_index, storage, user, ts):
    plt.pie([sales_potential_index, 5 - sales_potential_index], colors=['blue', 'white'], startangle=90)
    plt.title('Sales Potential', fontsize=20)
    plt.suptitle('\n\n\n\n Based on the recommendation Score, competition index and Best Selling Rank', fontsize=7)
    my_circle = plt.Circle((0, 0), 0.7, color='white')
    p = plt.gcf()
    pie = p.gca().add_artist(my_circle)
    p.savefig(ts + 'sales_chart.png')
    plt.close()
    url5 = getURL(ts + "sales_chart" + ".png", storage, user)
    return url5


def product_review_analysis(asin, storage, user, ts):
    # Processing Part
    data_raw = get_data_for_asin(asin)
    clean = clean_data(data_raw)
    all_reviews = wrangle_data(clean)
    summarised_data = summarise_data(all_reviews['review_set'], asin)
    for key in summarised_data.keys():
        all_reviews[key] = summarised_data[key]
    summary = all_reviews
    number_good = len(summary['good']['reviews'])
    number_bad = len(summary['bad']['reviews'])

    good_wordcloud = get_word_cloud(summary['good']['reviews'], ts + 'good_wc', storage, user, ts)
    bad_wordcloud = get_word_cloud(summary['bad']['reviews'], ts + 'bad_wc', storage, user, ts)
    remndnScore = get_recommendation_score(summary)
    scatter = get_scatter_plot(summary, storage, user, ts)
    pie_good_bad = get_pie_chart([number_good / (number_good + number_bad), number_bad / (number_good + number_bad)],
                                 ['Good Reviews', 'Bad Reviews'], storage, user, ts)

    return {'good_wordcloud': good_wordcloud, 'bad_wordcloud': bad_wordcloud,
            'recommendation_score': round(remndnScore, 2), 'scatter_plot': scatter,
            'total_reviews': number_good + number_bad, 'pie_good_bad': pie_good_bad, 'product_title': data_raw['title']}


def competition_analysis(asin, storage, user, ts):
    meta_data = get_meta_data(asin)

    similar_items = []
    soup = BeautifulSoup(meta_data['similar_item'], "html.parser")
    rows = soup.findChildren(['tr'])
    if len(rows) == 0:
        return {'Competition_Index': 5, 'bar_plot': "NULL"}
    rows = rows[3]
    data = rows.find_all('a')
    for t_row in data:
        link = t_row.get('href')
        similar = link.split('/')[2]
        similar = similar.strip()
        similar_items.append(similar)

    result = {}
    result['total_similar_objects'] = len(similar_items)
    similar_rec_score = {}
    for i in similar_items:
        data_raw = get_data_for_asin(i)

        try:
            title = data_raw['title']
            clean = clean_data(data_raw)
            all_reviews = wrangle_data(clean)
            remndnScore = get_recommendation_score(all_reviews)
        except:
            continue

        similar_rec_score[title] = remndnScore
    result['similar_objects_with_sufficient_data'] = len(similar_rec_score)

    similar = pd.DataFrame.from_dict(similar_rec_score, orient='index')
    similar['names'] = similar.index
    similar['recommendation_score'] = similar.iloc[:, 0]
    similar = similar.drop(columns=0)
    if len(similar) <= 1:
        return {'Competition_Index': 5, 'bar_plot': "NULL"}
    similar['CompetitionIndex'] = 5 * ((similar['recommendation_score'] - min(similar['recommendation_score'])) / (
            max(similar['recommendation_score']) - min(similar['recommendation_score'])))
    result['bar_plot'] = get_bar_plot(similar, storage, user, ts)
    result['Competition_Index'] = round(similar['CompetitionIndex'][0], 2)
    return result


def sales_potential(asin, product_rvw, competition, storage, user, ts):
    meta_data = get_meta_data(asin)
    rank_txt = meta_data['rank']
    if len(rank_txt) == 1:
        rtxt = rank_txt[0]
    else:
        rtxt = rank_txt[1]
    rank = get_rank_from(rtxt, meta_data['main_cat'])
    rankIndex = (((rank[1] - rank[0]) / rank[1]) * 5)
    sales_potential_index = round(
        (rankIndex + competition['Competition_Index'] + product_rvw['recommendation_score']) / 3, 2)
    potn_chart = get_sales_pot_chart(sales_potential_index, storage, user, ts)

    return {'sales_potential_index': sales_potential_index, 'bsr': rank[0], 'total_products_in_category': rank[1],
            'sales_chart': potn_chart}
