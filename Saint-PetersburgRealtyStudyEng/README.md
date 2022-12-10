# Saint-Petersburg's realty market study

We have access to archive data from the [Yandex.Realty](https://realty.ya.ru/) service on apartments in St. Petersburg and neighboring settlements. The key objective is to learn how to determine market values of real estate. Study reslults might help to build an automated system to track anomalies and realty frauds.
Each real estate ad implies two data sources. Some of these data were entered by service users: total square footage, living area square footage, kitchen  square footage, number of rooms and balconies, price and so on. Others were obtained automatically with Yandex GIS: distances to a settlement center, airport, nearest park or pond.

**Our project is aimed** to identify key parameters determining real estate market values.

In the course of the project we:

1. Performed preliminary data analysis. 
2. Conducted exploratory data analysis (EDA) and preprocessed the data:
    - identified missing values and suggested possible explanations for missing data;
    - filled in the missing values, if necessary;
    - calculateed several synthetic features: square meter price, day of week, month and year when an ad was created, floor categories, proportions of square footage for different apartment parts.
3. Performed correlation analysis, graphic trend analysis.

We evaluated key parameters that determine the market value of realty in St. Petersburg and the Leningrad oblast (region). Data distributions by price, area, number of rooms, and ceiling heights have been studied. A fairly reasonable hypothesis about a direct connection between the property price and its square footage (which, in turn, is related to the number of rooms) has been confirmed. 

In addition, the distribution of the ads by different indicators provides grounds for some conclusions about the typical consumer behavior of real estate sellers:
- Ads are posted mainly on weekdays.
- The most expensive (by median price) property is posted Tuesdays and Wednesdays.
- The most "expensive" month in real estate supply is April, and the cheapest is June.
- Generally, the period from February to April is the most active real estate sales.

Python libraries used: pandas, Matplotlib, Phi_K, seaborn, skimpy, statsmodels, pymystem3 