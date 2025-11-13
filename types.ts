
export interface NutritionInfo {
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
}

export interface DetectedFood {
  name: string;
  portionSize: string;
  nutrition: NutritionInfo;
}

export interface LoggedMeal {
  id: string;
  food: DetectedFood;
  portions: number;
  date: string; // ISO string
}

export type View = 'home' | 'analysis' | 'history';
