from django.shortcuts import redirect, render
from . import views
from .models import *
from .forms import *

# Create your views here.

def simulation(request): #Ecran_1
    if request.method == 'POST':
        form = SimulationForm(request.POST)
        if form.is_valid():
            amount = form.cleaned_data['amount']
            duration = form.cleaned_data['duration']
            
            request.session["datasim"]= {"amount": amount, "duration": duration} #enregistrement des données de champs dans la session active

            return redirect('/simulation/pro_situation') #Si les champs sont validés on passe au prochain écran
    else:
        form = SimulationForm()
    return render(request, 'simulation/simulation.html', {'form': form}) #Si les champs ne sont pas valides le meme ecran se recharge

def pro_situation(request): #Ecran_2
    if request.method == 'POST':
        form = ProSituationForm(request.POST)
        if form.is_valid():
            situation = form.cleaned_data['situation']
            sector = form.cleaned_data['sector']
            profession = form.cleaned_data['profession']
            hire_date = form.cleaned_data['hire_date']
            contract_type = form.cleaned_data['contract_type']
               
            datasim = request.session.get("datasim") #chargement des données précédement chargées en session

            #maj du dict de session active
            datasim["situation"] = situation
            datasim["sector"] = sector
            datasim["profession"] = profession
            datasim["hire_date"] = hire_date
            datasim["contract_type"] = contract_type

            #enregistrement des données de champs de 2 écrans dans la session active
            request.session["datasim"] = datasim
            return redirect('/simulation/applicant')
    else:
        form = ProSituationForm()
    return render(request, 'simulation/pro_situation.html', {'form': form}) #Si les champs ne sont pas valides le meme ecran se recharge

def applicant(request): #Ecran_3
    if request.method == 'POST':
        form = ApplicantForm(request.POST)
        if form.is_valid():
            lname = form.cleaned_data['lname']
            #Mettre les autres champs 
            applicant = Applicant()
            applicant.last_name = lname
            applicant.save() #ecriture vers la BD
            #datasim = request.session.get("datasim")  Cette 

            #request.session["applicant"]= saved_applicant #Verification Id dans DB ///A VERIFIER
            return redirect('/simulation/charge')
    else:
        form = ApplicantForm()
    return render(request, 'simulation/applicant.html', {'form': form})


def charge(request): #Ecran_4

    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = ChargeForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            real_estate_loan = form.cleaned_data['real_estate_loan']
            charge_applicant = Charge_Applicant()
            charge_applicant.real_estate_loan = real_estate_loan
            ######################################################
        
            active_loan = form.cleaned_data['active_loan']
            charge_applicant.active_loan = active_loan
            loan_type = form.cleaned_data['loan_type']
            charge_applicant.loan_type = loan_type
            monthly_amount = form.cleaned_data['monthly_amount']
            charge_applicant.monthly_amount = monthly_amount
            applicant = request.session.get("applicant") #Verification session precedente Applicant (cette ligne est a reverifier)
            charge_applicant.applicant = applicant   
            charge_applicant.save()
        return redirect('/simulation/project')

    # if a GET (or any other method) we'll create a blank form
    else:
        form = ChargeForm()
    return render(request, 'simulation/charge.html', {'form': form})


def calculateResult(request):
    # une requete de la bdd pour chercher les tables en relation avec l'id de l'applicant

    # calculer le resultat final
    # retourner la pager de resultat avec le resultat
    return