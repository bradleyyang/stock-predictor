from fastapi import FastAPI, HTTPException
from utils import get_stock, get_predictions
from pydantic import BaseModel

app = FastAPI()

class Stock_Inputs(BaseModel):
    ticker: str
    data_periods: str
    data_intervals: str
    prediction_periods: int
    prediction_intervals: str

@app.post("/stockprices")
def get_stockprices(stock_inputs: Stock_Inputs):
    ticker, data_periods, data_intervals, prediction_periods, prediction_intervals = stock_inputs.model_dump().values()
    try:
        stock_data = get_stock(ticker, data_periods, data_intervals)
        if stock_data is None:
            return {"error": "stock data is none"}
        if stock_data.empty:
            return {"error": "No data found for stock"}
        predictions = get_predictions(prediction_periods, prediction_intervals, stock_data)
        if predictions is None:
            return {"error": "predictions is none"}
        return {"ticker": ticker, "predictions": predictions}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error: " + str(e))