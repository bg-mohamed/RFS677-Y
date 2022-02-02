from django.db import models
# Create your models here.
# definir l'objet et le lier avec la bdd 


class Applicant(models.Model) : #A COMPLETER par IBRA
    last_name =  models.CharField(max_length=30, null=True)

class Pro_Situation(models.Model) :
    situation =  models.CharField(max_length=30,null=True)
    sector = models.CharField(max_length=30,null=True)
    profession = models.CharField(max_length=30,null=True)
    hire_date = models.CharField(max_length=30,null=True)
    contract_type = models.CharField(max_length=30,null=True)
    applicant = models.ForeignKey(Applicant, on_delete=models.CASCADE, null=True)


class Charge_Applicant(models.Model) :
    real_estate_loan =  models.BooleanField(null=True)
    active_loan = models.IntegerField(null=True)
    loan_type = models.CharField(max_length=30,null=True)
    monthly_amount = models.DecimalField(decimal_places=5,max_digits=10,null=True)
    applicant = models.ForeignKey(Applicant, on_delete=models.CASCADE, null=True)


