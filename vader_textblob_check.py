################################Vader Demo#####################################

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()
print(analyser.polarity_scores("This is a good course"))
print(analyser.polarity_scores("This is an awesome course")) # degree modifier
print(analyser.polarity_scores("The instructor is so cool"))
print(analyser.polarity_scores("The instructor is so cool!!") )# exclaimataion changes score
print(analyser.polarity_scores("The instructor is so COOL!!")) # Capitalization changes score
print(analyser.polarity_scores("Machine learning makes me :)")) #emoticons
print(analyser.polarity_scores("His antics had me ROFL"))
print(analyser.polarity_scores("The movie SUX")) #Slangs)

################################Textblob Demo##################################
print("printing the scores for textblob")
from textblob import TextBlob

print(TextBlob("His").sentiment)
#print(TextBlob("remarkable").sentiment)
print(TextBlob("work").sentiment)
print(TextBlob("ethic").sentiment)
print(TextBlob("impressed").sentiment)
print(TextBlob("me").sentiment)
print(TextBlob("His remarkable work ethic impressed me").sentiment)