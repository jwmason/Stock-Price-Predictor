import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class StockService {
  
  private apiUrl = 'http://127.0.0.1:5000';  // Backend URL

  constructor(private http: HttpClient) {}

  getPrediction(stockSymbol: string): Observable<any> {
    let params = new HttpParams().set('symbol', stockSymbol);
    return this.http.get(`${this.apiUrl}/predict`, { params });
  }
}
