from django import forms


class ApplicantForm(forms.Form):
    lname = forms.CharField()

class ChargeForm(forms.Form):
    real_estate_loan = forms.BooleanField()
    active_loan = forms.IntegerField()
    loan_type = forms.CharField()
    monthly_amount = forms.DecimalField()

class SimulationForm(forms.Form):
    amount = forms.FloatField()
    duration = forms.FloatField()

class ProSituationForm(forms.Form):
    situation = forms.CharField()
    sector = forms.CharField()
    profession = forms.CharField()
    hire_date = forms.CharField()
    contract_type = forms.CharField()