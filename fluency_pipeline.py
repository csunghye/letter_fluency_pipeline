import spacy
from scipy import spatial
from pydub import AudioSegment
from dtw import *
import librosa
import numpy as np
import pandas as pd
import argparse, glob, os
import nltk

nlp = spacy.load('en_core_web_lg')

def correct_word(word, LETTER):
	if word.lower().startswith(LETTER) & (not word.endswith("-")):
		return True
	else:
		return False

def get_wordVec(WORDVEC):
    f = open(WORDVEC,'r')
    embeddings_dict = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = [value for value in splitLines[1:]]
        embeddings_dict[word] = wordEmbedding
    return embeddings_dict

def get_mfcc(data_wav, start, end, fs):
	wav_crop = data_wav[round(float(start)*1000) : round(float(end)*1000)]
	wav_crop_samples = wav_crop.get_array_of_samples()
	mfcc_val = librosa.feature.mfcc(y=np.array(wav_crop_samples).astype(np.float32), sr=fs, n_mfcc=13)
	mfcc_transpose = np.transpose(mfcc_val)
	return mfcc_transpose

def read_audio(filename, audio_dir):
	audio_name = filename.split('/')[-1]+'.wav'
	data_wav = AudioSegment.from_wav(audio_dir+'/'+audio_name)
	fs = np.array(data_wav.frame_rate)
	return data_wav, fs

def get_phon_sim(prev_mfcc, mfcc):
	if np.isnan(prev_mfcc).all() != True :
		phon_sim = dtw(prev_mfcc, mfcc, keep_internals=True, step_pattern='asymmetric',window_args = {'window_size':100}, open_end=True).normalizedDistance
	else:
		phon_sim = np.nan
	return phon_sim

def get_sem_sim(embeddings_dict, word, prev_word):
	if word in embeddings_dict and prev_word in embeddings_dict and prev_word != 'NA':
		sem_sim = spatial.distance.euclidean(np.asarray(embeddings_dict[word], "float32"), np.asarray(embeddings_dict[prev_word], "float32"))
		prev = word
	else:
		sem_sim = np.nan
	return sem_sim

def get_dur_pos_phon_sem(file, data_wav, embeddings_dict, fs, LETTER):
	FILLERS = ['um','uh','eh']
	BACKCHANNELS = ['hm', 'yeah', 'mhm', 'huh']

	newdf = []
	prev_mfcc = np.array([], dtype=np.float64)
	infile = open(file, 'r')
	pause_dur = 0
	order = 0
	prev_word = 'NA'
	phon_sim = 0

	for line in infile:
		if not (line.startswith('\\')):
			data = line.rstrip('\n').split('\t')
			if len(data) > 2:
				if correct_word(data[2], LETTER) == True:
					word_dur = float(data[1]) - float(data[0])
					doc = nlp(data[2])
					if doc[0].pos_ != "NUM" and doc[0].pos_ != "PROPN" and data[2].lower() != prev_word:
						mfcc = get_mfcc(data_wav, data[0], data[1], fs)
						phon_sim = get_phon_sim(prev_mfcc, mfcc)
						sem_sim = get_sem_sim(embeddings_dict, data[2].lower(), prev_word)
						order +=1
						newdf.append([file, data[0], data[1], data[2].lower(), order, word_dur, pause_dur, doc[0].pos_, doc[0].lemma_, phon_sim, sem_sim])
						prev_mfcc = mfcc
						prev_word = data[2].lower()
						word_dur = 0
						pause_dur = 0
						
				else: 
					word_dur = float(data[1]) - float(data[0])
					pause_dur += word_dur
					
			else:
				word_dur = float(data[1]) - float(data[0])
				pause_dur += word_dur
					

	df = pd.DataFrame(newdf, columns=['file','start','end','word','order','word_dur','prev_pause_dur','POS','lemma', 'phon_sim', 'sem_sim'])
	
	return df

def count_f_words(df, LETTER):
	all_f_words = df.word.str.lower().str.startswith(LETTER).sum()
	propn = df.POS.str.startswith("PROPN").sum()
	num = df.POS.str.startswith("NUM").sum()
	repetition = df.duplicated('lemma').sum()
	return all_f_words, propn, num, repetition

def add_lexical(df, phonDf, measureDict):
	word_lexical = pd.merge(df, measureDict, on='word', how='left')
	lemma_lexical = pd.merge(df, measureDict, left_on='lemma', right_on='word', how='left')
	df = word_lexical.fillna(lemma_lexical)
	df = pd.merge(df, phonDf[["word", "phon", "syll"]], on='word', how='left')
	return df

def get_phondict():
	phonDf = pd.DataFrame.from_dict(nltk.corpus.cmudict.dict(), orient='index')
	phonDf = phonDf.reset_index()
	phonDf['phon'] = phonDf[0].map(len)
	phonDf = phonDf.drop(columns=[1,2,3,4])
	phonDf.columns = ['word','pron','phon']
	phonDf['pronstring'] = [','.join(map(str, l)) for l in phonDf['pron']]
	phonDf['syll'] = phonDf.pronstring.str.count("0|1|2")
	return phonDf

def main(args):
	LEXICAL_LOOKUP = os.path.join(os.path.dirname(__file__),'all_measures_raw.csv')
	LETTER = args.letter
	AUDIO_DIR = args.audio_folder
	WORDVEC = args.glove_folder +'/glove.6B.300d.txt'

	measureDict = pd.read_csv(LEXICAL_LOOKUP)
	phonDf = get_phondict()
	embeddings_dict = get_wordVec(WORDVEC)
	allResults = pd.DataFrame()
	if args.FA_filetype:
		filelist = glob.glob(args.FA_folder+'/'+args.FA_filetype)
	else:
		filelist = glob.glob(args.FA_folder+'/*.word')

	with open(args.score_file, 'w') as outFile:
		outFile.writelines('filename,total_words,proper_noun,numerics,repetitions\n')
		for file in filelist:
			print(file, " is being processed...")
			filename = file.split('.')[0]
			data_wav, fs = read_audio(filename, AUDIO_DIR)
			
			df = get_dur_pos_phon_sem(file, data_wav, embeddings_dict, fs, LETTER)
			score, propn, number,repetition = count_f_words(df, LETTER)
			df_lexical = add_lexical(df, phonDf, measureDict)
			allResults = pd.concat([allResults, df_lexical], sort=True)
			outFile.writelines(file+','+str(score)+','+str(propn)+','+str(number)+','+str(repetition)+'\n')
	allResults.to_csv(args.measure_file, index=False)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-score_file', type=str, required=True, help='Name the output file for scores')
	parser.add_argument('-measure_file', type=str, required=True, help='Name the output file for measures')
	parser.add_argument('-audio_folder', type=str, required=True, help='Folder containing input audio files')
	parser.add_argument('-FA_folder', type=str, required=True, help='Folder containing forced alignment outputs')
	parser.add_argument('-FA_filetype', type=str, required=False, help='File type of FA outputs')
	parser.add_argument('-letter', type=str, required=True, help='Letter for the fluency task')
	parser.add_argument('-glove_folder', type=str, required=True, help='Location of glove wordvec files')
	args = parser.parse_args()
	main(args)
	
	
	




