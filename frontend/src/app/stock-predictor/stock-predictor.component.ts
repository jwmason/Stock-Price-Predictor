import { Component } from '@angular/core';
import { StockService } from '../stock.service';

@Component({
  selector: 'app-stock-predictor',
  templateUrl: './stock-predictor.component.html',
  styleUrls: ['./stock-predictor.component.css']
})
export class StockPredictorComponent {
  stockSymbol: string = '';
  prediction: any;
  loading: boolean = false;
  error: string = '';

  constructor(private stockService: StockService) {}

  onSubmit() {
    this.loading = true;
    this.error = '';
    this.stockService.getPrediction(this.stockSymbol)
      .subscribe(
        data => {
          this.prediction = data;
          this.loading = false;
        },
        error => {
          this.error = 'Error fetching prediction';
          this.loading = false;
        }
      );
  }
}
