from typing import Union
from fastapi import FastAPI, HTTPException
from utils import get_stock, get_predictions
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/stockprices/{stock}")
def read_stockprices(stock: str):
    try:
        stock_data = get_stock(stock, "1mo", "1d")
        if stock_data is None:
            return {"error": "stock data is none"}
        if stock_data.empty:
            return {"error": "No data found for stock"}
        predictions = get_predictions(10, "B", stock_data)
        if predictions is None:
            return {"error": "predictions is none"}
        return {"stock": stock, "predictions": predictions}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error: " + str(e))