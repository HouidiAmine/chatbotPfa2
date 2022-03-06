
import nltk 
from  nltk.chat.util  import Chat,reflections
pairs=[ 
   ["hello c est (.*)",["hello %1 \ncomment je peux vous aider" ]],
   ["(bjr|bnj|bonjour|coco|hi|good afternoon)(.*)",["salut! comment je peux vous aider?"]],
   ["(sava|cv|how are you |what's up)",["oui cv ",]],
   ["(.*) (.*) (.*)taille (.*)",["ok je vais te preparer %2 %3 taille %4 passez votre commande sur notre site "]],
   ["(.*)(adresse|ville|localisation|ou(.*)boutique)",["nous sommes disponible a tunis rue hbib bourguiba "]],
   ["(.*)(aide|help)(.*)",["je peux vous aider","c est un plaisir pour vous aider"]],
   ["(.*)merci(.*)",["bienvenu...a bientot"]],
   ["(.*)couleur (.*)(.*)",["contactez le site pour le couleur %2 "]]
    ]
chat=Chat(pairs,reflections)
chat.converse()





 