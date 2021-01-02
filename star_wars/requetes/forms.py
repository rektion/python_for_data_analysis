from django import forms


class ChoiceForm(forms.Form):     
     myBoolField = forms.BooleanField(
        label='Afficher les résultats approfondis ?',
        initial=False
     )