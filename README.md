# Dal libro Arithmós 
*Tempo · Relatività · Quanti · Intelligenze
Dai pitagorici alla coscienza artificiale*

- [MondadoriStore](https://www.mondadoristore.it/arithmos-tempo-relativita-quanti-intelligenze-dai-pitagorici-alla-coscienza-artificiale-libro-luigi-belcamino/p/9791224076780 "MondadoriStore")
- [Amazon](https://amzn.eu/d/0gmjyQnW "Amazon")
- [Youcanprint Store](https://store.youcanprint.it/arithmos-tempo-relativita-quanti-intelligenze-dai-pitagorici-alla-coscienza-artificiale/b/d4089bb3-c7ba-5343-921f-9c353ddee1ff "Youcanprint Store")
 

## Esercitazione Ex3.inferenza
Si utilizza un LLM locale (scaricato da Hugging Face), si configura la quantizzazione e si esegue inferenza.
Il codice *Python* scaricherà in locale un modello LLM da Hugging Face e gli porrà una domanda nel formato atteso dal modello stesso.
Come primo passo viene verificata la disponibilità di una GPU (CUDA) per ottimizzare le prestazioni, altrimenti sarà utilizzata la CPU.

Successivamente si installano le librerie essenziali per le funzioni di transformer, tokenizer e quantizzazione. 

Poi si procede al download del modello: per poterlo fare occorre essersi preventivamente registrati sul sito per ottenere un token di autenticazione e poi averlo memorizzato in Google Colab, sezione Secret con il nome HF_TOKEN. 

La tecnica di quantizzazione a quattro bit è utilizzata per contenere il consumo di memoria. 

Il modello utilizzato è ``Mistral-7B-Instruct-v0.2``: ha sette miliardi di parametri (pesi di addestramento) ed è instruction tuned, ovvero è specificamente addestrato a seguire le istruzioni; il prompt che gli viene fornito, infatti, include i ruoli (*user* identifica l’utente, *assistant* identifica il modello) che aiutano il modello a capire il contesto.
