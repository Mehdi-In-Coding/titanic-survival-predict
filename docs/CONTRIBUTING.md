💡 Guide de Contribution

voici les manip pour push vos modif à chaque fois :

🚀 Étape 1 : Forker le Dépôt

Allez sur le dépôt GitHub : https://github.com/votre-github/titanic-survival-predict

Cliquez sur Fork.

Clonez votre fork sur votre machine :

git clone https://github.com/votre-utilisateur/titanic-survival-predict.git
cd titanic-survival-predict

🛠️ Étape 2 : Créer une Branche

git checkout -b feature/nom-de-votre-fonctionnalite

✍️ Étape 3 : Ajouter et Tester votre Code

Ajoutez votre nouvelle fonctionnalité ou corrigez un bug.

Assurez-vous que votre code suit le style PEP8.

Exécutez les tests unitaires :

pytest tests/

✅ Étape 4 : Commit et Push

git add .
git commit -m "Ajout de [nom de la fonctionnalité]"
git push origin feature/nom-de-votre-fonctionnalite

🔄 Étape 5 : Ouvrir une Pull Request (PR)

Allez sur le dépôt GitHub.

Cliquez sur New Pull Request.

Sélectionnez votre branche et soumettez la PR.