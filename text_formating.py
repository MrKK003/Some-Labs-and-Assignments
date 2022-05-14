import time

class analysedText(object):
    
    def __init__ (self, text):
        
        
        # TODO: Remove the punctuation from <text> and make it lower case.
        
        self.text=text
        
        not_allowed_punctuations=".,!?"
        
        for character in not_allowed_punctuations:
            text = text.replace(character, "")
            
        self.text=text.lower()
        
        # TODO: Assign the formatted text to a new attribute called "fmtText"
        
        self.fmtText=self.text

        
        pass 
    
    def freqAll(self):    
        
        # TODO: Split the text into a list of words  
        
        words=self.fmtText.split()
             
        # TODO: Create a dictionary with the unique words in the text as keys
        # and the number of times they occur in the text as values
        dictionary={}
        
        for word in words:
            
            if (word not in dictionary.keys()):
                dictionary[word]=[]
                dictionary[word].append(words.count(word))
                
        print(dictionary)
                
      
        return dictionary # return the created dictionary
    
    def freqOf(self, word):
        
        dict1=self.freqAll() 
        
        # TODO: return the number of occurrences of <word> in <fmtText>
        if (word in dict1):
            return dict1.get(word)
        else:
            return 0
         

def main():
    text1="Nevertheless, the banks were finding it harder to obtain funds. Landsbanki then created Icesave, a saving account with an attractive interest rate for foreigners. Its branches in UK and Luxembourg witnessed a massive influx of deposits with high-profile customers such as Cambridge University, the London Metropolitan Police Authority, or the UK Audit Commission. Another risky financial instrument was the love letter. The bank issued bonds to its branches, which then used them as collateral to obtain funds from the Central Bank. They even used this dubious transaction in the foreign market, borrowing from the Luxembourg Central Bank and the European Central Bank."
    C1=analysedText(text1)
    print(C1.freqAll())
    print(C1.freqOf("the"))
    

if __name__ == "__main__": 
    t1=time.perf_counter()
    main()
    t2=time.perf_counter()
    print(f'Finished in {t2-t1} seconds')