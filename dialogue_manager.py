import os
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.metrics.pairwise import cosine_similarity
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from utils import *

def rank_candidates(q_emb, candidate_threads_emb, candidate_thread_ids):
    """
        q_emb: embedding vector for a question
        candidate_threads_emb: matrix of candidate thread embeddings which we want to rank
        candidate_thread_ids: list of stackoverflow thread ids aligned with candidate_threads_emb
        
        result: a list of sorted tuples (thread_id, cos_similiarity)
    """

    canditate_similarities = cosine_similarity(q_emb.reshape(1, -1), candidate_threads_emb)[0]
    sorted_candidates = sorted([(candidate_thread_ids[i], s) for (i, s) in enumerate(canditate_similarities)], key=lambda k: k[1], reverse=True)
    return sorted_candidates

class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings(paths['WORD_EMBEDDINGS'])
        self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".pkl")
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)

        # HINT: you have already implemented a similar routine in the 3rd assignment.
        
        question_vec = question_to_vec(text_prepare(question), self.word_embeddings, self.embeddings_dim)
        best_thread, cos_similarity = rank_candidates(question_vec, thread_embeddings, thread_ids)[0]
        print("Best thread_id: {} with cosine similarity: {}".format(best_thread, cos_similarity))
        return best_thread


class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources from...{}".format(paths))

        # Intent recognition:
        self.intent_recognizer = unpickle_file(paths['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = unpickle_file(paths['TFIDF_VECTORIZER'])

        self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])
        self.thread_ranker = ThreadRanker(paths)

        #chitchat bot
        self.chitchat_bot = self.create_chitchat_bot()

    def create_chitchat_bot(self):
        """Initializes self.chitchat_bot with some conversational model."""

        # Hint: you might want to create and train chatterbot.ChatBot here.
        # It could be done by creating ChatBot with the *trainer* parameter equals 
        # "chatterbot.trainers.ChatterBotCorpusTrainer"
        # and then calling *train* function with "chatterbot.corpus.english" param
        english_bot = ChatBot("SOChatterbot")
        trainer = ChatterBotCorpusTrainer(english_bot)
        trainer.train('chatterbot.corpus.english')
        # trainer.train("chatterbot.corpus.english.greetings")
        # trainer.train("chatterbot.corpus.english.conversations")

        return english_bot
       
    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.
        
        prepared_question = text_prepare(question)
        tfidf_features = self.tfidf_vectorizer.transform(np.array([prepared_question]))
        intent = self.intent_recognizer.predict(tfidf_features)[0]
        print("I think your intent is: {}".format(intent))

        # Chit-chat part:   
        if intent == 'dialogue':
            # Pass question to chitchat_bot to generate a response.       
            response =  self.chitchat_bot.get_response(question)
            return response
        
        # Goal-oriented part:
        else:        
            # Pass features to tag_classifier to get predictions.
            tag = self.tag_classifier.predict(tfidf_features)[0]
            
            print("I TAG your question as: {}".format(tag))
            # Pass prepared_question to thread_ranker to get predictions.
            thread_id = self.thread_ranker.get_best_thread(prepared_question, tag)
           
            return self.ANSWER_TEMPLATE % (tag, thread_id)

