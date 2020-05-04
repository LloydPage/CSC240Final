import pandas
def Preprocessing():
    data1=pandas.read_csv("Deathrate.csv")
    data2=pandas.read_csv("GdpPerCapita.csv")
    data3=pandas.read_csv("Inflation.csv")
    data5=pandas.read_csv("Unemployment.csv")
    data6=pandas.read_csv("UrbanPop.csv")
    templist=[[] for i in range(8)]
    devCountries=["Andorra", "Austria", "Belgium", "Czech Republic", "Denmark", "Estonia", "Faroe Islands", "Finland","France", "Germany","Greece","Guernsey","Holy See","Iceland","Iceland","Ireland","Italy","Jersey","Latvia","Liechtenstein","Lithuania","Luxembourg","Malta","Monaco","Netherlands","Norway","Portugal","San Marino","Slovak Republic","Slovenia","Spain","Sweden","Switzerland","United Kingdom","Cyprus","Hong Kong SAR, China","Macao SAR, China","Singapore","Israel","Korea, Rep.","Japan","Taiwan","Bermuda","Canada","United States","Puerto Rico","Australia","New Zealand"]
    z=0
    for y in range(len(data1.index)):
        for x in range (4, len(data1.columns)):
            templist[0].append(data1.iloc[y][0])
            templist[1].append(1956+x)
            templist[2].append(data1.iloc[y][x])
            templist[3].append(data2.iloc[y][x])
            templist[4].append(data3.iloc[y][x])
            templist[5].append(data5.iloc[y][x])
            templist[6].append(data6.iloc[y][x])
            #1 for developed (per IMF), 0 for developing (per IMF)
            if data1.iloc[y][0] in devCountries:
                if 1955+x>2015:
                    templist[7].append(1)
                elif 1955+x<2015 and data1.iloc[y][0]=="Lithuania":
                    templist[7].append(0)
                elif 1955+x<2014 and data1.iloc[y][0]=="Latvia":
                    templist[7].append(0)
                elif 1955+x<2011 and data1.iloc[y][0]=="Estonia":
                    templist[7].append(0)
                elif 1955+x<2009 and data1.iloc[y][0]=="Slovak Republic":
                    templist[7].append(0)
                elif 1955+x<2009 and data1.iloc[y][0]=="Czech Republic":
                    templist[7].append(0)
                elif 1955+x<2008 and data1.iloc[y][0]=="Malta":
                    templist[7].append(0)
                elif 1955+x<2007 and data1.iloc[y][0]=="Solvenia":
                    templist[7].append(0)
                elif 1955+x<2001 and data1.iloc[y][0]=="Cyprus":
                    templist[7].append(0)
                elif 1955+x<1997 and data1.iloc[y][0]=="Taiwan":
                    templist[7].append(0)
                elif 1955+x<1997 and data1.iloc[y][0]=="Korea, Rep.":
                    templist[7].append(0)
                elif 1955+x<1997 and data1.iloc[y][0]=="Singapore":
                    templist[7].append(0)
                elif 1955+x<1997 and data1.iloc[y][0]=="Israel":
                    templist[7].append(0)
                elif 1955+x<1997 and data1.iloc[y][0]=="Hong Kong SAR, China":
                    templist[7].append(0)
                else:
                    templist[7].append(1)
            else:
                templist[7].append(0)
            z=z+1
    templist=list(map(list, zip(*templist)))
    data=pandas.DataFrame(templist, columns=['Name','year','Death rate','GDP per capita','Inflation','Unemployment','Urban Population','Development Status'])
    droplist=[]
    for y in reversed(range(len(data.index))):
        for x in reversed(range(len(data.columns))):
            if pandas.isna(data.iloc[y][x]):
                droplist.append(y)
    data=data.drop(droplist)
    return data

def main():
    data=Preprocessing() #data should include properly cleaned data for use in the algorithms
if __name__=="__main__": 
    main() 
