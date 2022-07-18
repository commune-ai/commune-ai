
# Graph QL Query Schema


- To pull in token data and predictions we use graph ql for creating a flexible database
- At the moment you are able to query the following info
    - Historical Coin Data via Binance API
    - Predictions via Models
    

Use the following Schema

### Coin Ticker Data

**Input Schema**

    
      Name=graphene.String(default_value="LINKUSDT"),
      StartTime=graphene.DateTime(default_value=datetime.datetime.now() - datetime.timedelta(days=100)),
      EndTime=graphene.DateTime(default_value=datetime.datetime.now() - datetime.timedelta(days=40)))


**Output Schema**

    Name = String(required=True)
    StartTime = Date(required=False)
    EndTime = Date(required=False)


    # Direct Indicators (Add more if needed)
    Open = List(Float)
    Close = List(Float)
    High = List(Float)
    Low = List(Float)
    Volume = List(Float)
    NumberOfTrades = List(Float)
    TakerBuyBaseAssetVolume = List(Float)
    TakerBuyQuoteAssetVolume = List(Float)

    # Time Features
    Timestamp = List(Int)
    Date = List(DateTime)
**Example**

**Example**

Get ETHUSDT from 2021-08-18T15:00:00 to 2021-08-19T15:00:00

```angular2html
{ 
  token(Name:"ETHUSDT", 
        StartTime: ["2021-08-18T15:00:00"]
)
      {
          Open
          Close
          High
          Low
          Time
      }
}

```


### Prediction (Batch Inference Supported for Backtesting)

Querying predictions from the model. Current Taks are supported 
    - regression for Coin Pairs

**Input Schema**
```python

 # Name of the Coin
 CoinName=String(default_value="ETHUSDT"),

 # Filter Experiments/Runs Using this SpecifIC Model Key (Location in Model Directory)
 ModelName=String(default_value="complete.regression.GP_NbeatsTransformer"),
 
 # Filter Experiments Based on (MetricName) and specify Rank Order (Ascending)
 MetricName=String(default_value="val.MSE_future_Close"),
 Ascending=Boolean(default_value=True),
 
 # Specify a list of Start Times for the Prediction (For Batch Processing)
 StartTime=List(DateTime(default_value=datetime.datetime.now()))

 # Specify How Many Hours into the Future You Want
 ForcastPeriod=Int(default_value=48),
 
 # specify the experiment (experiments with a comma deliminator, like "DEMO,ALLAH")
 Experiments=List(String(default_value="DEMO"))
 
 # specify the experiment type and subtype (constant for now)
 Type=String(default_value="regression"),
 
 # subtype of the experiment
 SubType=String(default_value="crypto")

```

**Output Schema**
```angular2html
Mean = List(Float) # Mean Value 
Upper = List(Float) # Upper Bound Value
Lower = List(Float) # Lower Bound Value
TimeStamp = List(Int) # UTC timestamp
Date  = List(DateTime) # ISOFORMAT DateTime YYYY-MM-DDTHH:MM:SS

```

**Example**

The following query the prediction of ETHUSDT 48 hours into the future from the list of start times
[2021-08-18T15:00:00, 2021-08-20T15:00:00]



```

{
  predictions(
    					CoinName: "ETHUSDT",
    					ForcastPeriod: 48,
    					ModelName: "NBEATS", 
    					PredictionType: "Base", 
    					Experiment: ["DEMO"],
    					StartTime: ["2021-08-18T15:00:00", "2021-08-20T15:00:00"]
  ) {
    Mean
    Upper
    Lower
    Time
  }
}

```

