# Indian-Native-Language-Identification

Native Language Identi cation (NLI) is the process of identifying the native language of non-native speakers based on their speech or writing. It has several applications namely authorship pro ling and identi cation, forensic analysis, second language identi ca- tion, and educational applications. English is one of the prominent language used by most of the non-English people in the world. The native language of the non-English speakers may be easily identi ed based on their English accents. However, identi cation of native language based on the users posts and comments written in English is a challenging task.

I have implemented a supervised approach for this INLI task. The steps used in my approach are given below.
• Preprocess the given text
• Extract linguistics features for training data
• Build a neural network model from the features of training data
• Predict class label for the instance as any of the six languages namely Tamil, Hindi, Kannada, Malayalam, Bengali or Tel- ugu using the model

I have implemented our methodology in Python for the INLI task under FIRE 2017 Conference. The data set used to evaluate the task consists of a set of training data for six Indian languages and test data. The number of training instances are 207, 211, 203, 200, 202 and 210 for the languages Tamil, Hindi, Kannada, Malayalam, Bengali and Telugu respectively and number of test instances are 783. 

