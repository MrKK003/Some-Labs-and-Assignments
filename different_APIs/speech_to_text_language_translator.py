import time
import json
from ibm_watson import SpeechToTextV1 
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from pandas import json_normalize
from ibm_watson import LanguageTranslatorV3

def speech_to_text():
    url_s2t = "https://api.eu-gb.speech-to-text.watson.cloud.ibm.com/instances/0c6b2695-931c-406a-ab11-44f1a2e6cc81"
    iam_apikey_s2t = "sGBecrTIQuSKzmhzPt8s0Y10ojTWSSuey6QF_nizn0NX"

    authenticator = IAMAuthenticator(iam_apikey_s2t)
    s2t = SpeechToTextV1(authenticator=authenticator)
    s2t.set_service_url(url_s2t)

    filename='/Users/kirillkiptyk/PolynomialRegressionandPipelines.mp3'
    
    with open(filename, mode="rb")  as wav:
        response = s2t.recognize(audio=wav, content_type='audio/mp3')
    
    #print(response.result)
    json_normalize(response.result['results'],"alternatives")
    #print(response)
    recognized_text=response.result['results'][0]["alternatives"][0]["transcript"]
    #print(recognized_text)
    return recognized_text
    
def language_translator(recognized_text): 
    url_lt='https://api.eu-gb.language-translator.watson.cloud.ibm.com/instances/66064d92-fcf8-4c2f-8dac-1fa9be971cb2'
    apikey_lt='n4Mmclv6SJ69nWf5Un6MOi8Qz5mQbqBmCGhjjaN6xGEp'
    version_lt='2018-05-01'
    
    authenticator = IAMAuthenticator(apikey_lt)
    language_translator = LanguageTranslatorV3(version=version_lt,authenticator=authenticator)
    language_translator.set_service_url(url_lt)
    
    # languages = language_translator.list_languages().get_result()
    # print(json.dumps(languages, indent=2))
    
    json_normalize(language_translator.list_identifiable_languages().get_result(), "languages")
    
    translation_response = language_translator.translate(text=recognized_text, model_id='en-es')
    translation=translation_response.get_result()
    spanish_translation =translation['translations'][0]['translation']
    print(spanish_translation)
    
    translation_response = language_translator.translate(text=recognized_text, model_id='en-fr')
    translation=translation_response.get_result()
    french_translation =translation['translations'][0]['translation']
    print(french_translation)
    

def main():
    #recognized_text=[]
    recognized_text = speech_to_text()
    language_translator(recognized_text)


if __name__ == "__main__": 
    t1=time.perf_counter()
    main()
    t2=time.perf_counter()
    print(f'Finished in {t2-t1} seconds')