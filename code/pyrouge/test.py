from pythonrouge.pythonrouge import Pythonrouge

ROUGE_path = '/Users/gohan/Desktop/Thesis/pythonrouge/pythonrouge/RELEASE-1.5.5/ROUGE-1.5.5.pl' #ROUGE-1.5.5.pl
data_path = '/Users/gohan/Desktop/Thesis/pythonrouge/pythonrouge/RELEASE-1.5.5/data' #data folder in RELEASE-1.5.5

# initialize setting of ROUGE, eval ROUGE-1, 2, SU4, L
rouge = Pythonrouge(n_gram=2, ROUGE_SU4=True, ROUGE_L=True, stemming=True, stopwords=True, word_level=True, length_limit=True, length=50, use_cf=False, cf=95, scoring_formula="average", resampling=True, samples=1000, favor=True, p=0.5)

# system summary & reference summary
summary = [[" Tokyo is the one of the biggest city in the world."]]
reference = [[["The capital of Japan, Tokyo, is the center of Japanese economy."]]]

# If you evaluate ROUGE by sentence list as above, set files=False
setting_file = rouge.setting(files=False, summary=summary, reference=reference)

# If you need only recall of ROUGE metrics, set recall_only=True
result = rouge.eval_rouge(setting_file, recall_only=True, ROUGE_path=ROUGE_path, data_path=data_path)
print(result)
