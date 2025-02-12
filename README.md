# **Replication Guide for**  
 *Increased Persistence of Warm and Wet Winter Weather in Northwestern Europe Due to Trends Towards Strongly Positive NAO*

## ** Overview**
This guide explains how to **replicate** the results of our study.  
Follow the steps below to set up the necessary **data** and **code**.

---

##  1 Download & Set Up the Repository 
First, ensure you have the **WeatherPersistence** repository on your local device.  
If you haven't cloned it yet, run:
```bash
git clone git@github.com:bspanjers/WeatherPersistence.git
```
##  2 Download Weather Station Data 
The required weather station data is **too large** to be uploaded directly.  

 **Download the following datasets manually from**:  
 **[ECA&D Predefined Series](https://www.ecad.eu/dailydata/predefinedseries.php)**  

 You need these two datasets:
1. **Daily Precipitation Amount (RR)**
2. **Daily Mean Temperature (TG)**

 Click on the dataset name, and the files will **download automatically**.

---

##  3 Organize the Data Files
Once downloaded, **move the files** into the correct directory:  

### Move the data to the persistence_data folder
```bash
mv /path/to/downloaded/ECA_blend_rr/ persistence/data_persistence/
mv /path/to/downloaded/ECA_blend_tg/ persistence/data_persistence/
```

