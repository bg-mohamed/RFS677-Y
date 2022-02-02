# Generated by Django 2.2.5 on 2022-01-30 21:48

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Applicant',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('last_name', models.CharField(max_length=30, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Pro_Situation',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('situation', models.CharField(max_length=30, null=True)),
                ('sector', models.CharField(max_length=30, null=True)),
                ('profession', models.CharField(max_length=30, null=True)),
                ('hire_date', models.DateTimeField(null=True)),
                ('contract_type', models.CharField(max_length=30, null=True)),
                ('applicant', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='simulation.Applicant')),
            ],
        ),
        migrations.CreateModel(
            name='Charge_Applicant',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('real_estate_loan', models.BooleanField(null=True)),
                ('active_loan', models.IntegerField(null=True)),
                ('loan_type', models.CharField(max_length=30, null=True)),
                ('monthly_amount', models.DecimalField(decimal_places=5, max_digits=10, null=True)),
                ('applicant', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='simulation.Applicant')),
            ],
        ),
    ]
