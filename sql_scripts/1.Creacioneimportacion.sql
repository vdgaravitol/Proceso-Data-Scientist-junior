--Creación BD Fraude
DROP DATABASE IF EXISTS "Fraude";
CREATE DATABASE "Fraude"
     WITH
     OWNER = postgres
     ENCODING = 'UTF8'
	 TEMPLATE = template0
     LC_COLLATE = 'Spanish_Colombia.1252'
     LC_CTYPE = 'Spanish_Colombia.1252'
     TABLESPACE = pg_default
     CONNECTION LIMIT = -1
     IS_TEMPLATE = False;
	 
-- Creación tabla fraudes dentro de BD Fraude
	 
create table public.fraudes(
Monthh text,	
WeekOfMonth int,
DayOfWeek	text,
Make	text,
AccidentArea    text,	
DayOfWeekClaimed	text,
MonthClaimed	text,
WeekOfMonthClaimed  int,
Sex text,
MaritalStatus   text,	
Age	int,
Fault	text,
PolicyType	text,
VehicleCategory	text,
VehiclePrice	text,
FraudFound_P	int,
PolicyNumber	int,
RepNumber	int,
Deductible	int,
DriverRating	int,
Days_Policy_Accident	text,
Days_Policy_Claim	text,
PastNumberOfClaims	text,
AgeOfVehicle	text,
AgeOfPolicyHolder	text,
PoliceReportFiled	text,
WitnessPresent	text,
AgentType	text,
NumberOfSuppliments	   text,
AddressChange_Claim text,
NumberOfCars	text,
Yearr   int,
BasePolicy  text
)

--Inserción de datos desde el archivo local 'C:\GrupoR5\fraud.csv'
COPY PUBLIC.fraudes(Monthh, WeekOfMonth, DayOfWeek, Make, AccidentArea, DayOfWeekClaimed, MonthClaimed, WeekOfMonthClaimed, Sex, MaritalStatus, Age, Fault, PolicyType, VehicleCategory, VehiclePrice, FraudFound_P, PolicyNumber, RepNumber, Deductible, DriverRating, Days_Policy_Accident, Days_Policy_Claim, PastNumberOfClaims, AgeOfVehicle, AgeOfPolicyHolder, PoliceReportFiled, WitnessPresent, AgentType, NumberOfSuppliments, AddressChange_Claim, NumberOfCars, Yearr, BasePolicy)
FROM 'C:\GrupoR5\fraud.csv' DELIMITER ',' CSV HEADER;

--Salida 
SELECT   DISTINCT 
		monthh
		,weekofmonth
		,dayofweek
		,CAST(100*AVG(fraudfound_p) OVER (PARTITION BY monthh) AS DECIMAL(8,2))  AS "percentage_fraud_month"
		,CAST(100*AVG(fraudfound_p) OVER (PARTITION BY monthh, weekofmonth) AS DECIMAL(8,2)) AS "percentage_fraud_month_week"
		,CAST(100*AVG(fraudfound_p) OVER (PARTITION BY monthh, weekofmonth,dayofweek) AS DECIMAL(8,2)) AS "percentage_fraud_month_day"
FROM fraudes
ORDER BY monthh,weekofmonth

