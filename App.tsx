// FIX: Replaced placeholder content with a functional root App component.
import React, { useState, useEffect, useRef } from 'react';
import { DAILY_CALORIE_GOAL } from './constants';
import { estimateNutritionFromImage, getDailyTip } from './services/geminiService';
import { DetectedFood, LoggedMeal, View } from './types';
import { HomeIcon, HistoryIcon, CameraIcon, PlusIcon, MinusIcon, ChevronLeftIcon, TrashIcon, SunIcon, MoonIcon, ExportIcon } from './components/Icons';
import Spinner from './components/Spinner';

const App: React.FC = () => {
  const [view, setView] = useState<View>('home');
  const [meals, setMeals] = useState<LoggedMeal[]>([]);
  const [dailyTip, setDailyTip] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [theme, setTheme] = useState(localStorage.getItem('theme') || 'light');
  
  const [detectedFood, setDetectedFood] = useState<DetectedFood | null>(null);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [portions, setPortions] = useState(1);
  const [expandedMealId, setExpandedMealId] = useState<string | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // Theme management
  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prevTheme => (prevTheme === 'light' ? 'dark' : 'light'));
  };

  // Load meals from local storage on mount
  useEffect(() => {
    try {
      const storedMeals = localStorage.getItem('loggedMeals');
      if (storedMeals) {
        setMeals(JSON.parse(storedMeals));
      }
    } catch (e) {
      console.error("Failed to load meals from local storage", e);
    }
  }, []);

  // Save meals to local storage on change
  useEffect(() => {
    try {
      localStorage.setItem('loggedMeals', JSON.stringify(meals));
    } catch (e) {
      console.error("Failed to save meals to local storage", e);
    }
  }, [meals]);

  // Fetch daily tip on mount
  useEffect(() => {
    getDailyTip().then(setDailyTip).catch(console.error);
  }, []);

  const handleImageUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsLoading(true);
    setError(null);
    setDetectedFood(null);
    setView('analysis');

    const reader = new FileReader();
    reader.onloadend = async () => {
      const base64String = reader.result?.toString().split(',')[1];
      if (base64String) {
        setSelectedImage(reader.result as string);
        try {
          const food = await estimateNutritionFromImage(base64String);
          setDetectedFood(food);
          setPortions(1);
        } catch (err) {
          setError(err instanceof Error ? err.message : "An unknown error occurred.");
        } finally {
          setIsLoading(false);
        }
      } else {
        setError("Could not read the image file.");
        setIsLoading(false);
      }
    };
    reader.readAsDataURL(file);
    event.target.value = ''; // Reset file input
  };
  
  const handleLogMeal = () => {
    if (!detectedFood) return;
    const newMeal: LoggedMeal = {
      id: new Date().toISOString() + Math.random(),
      food: detectedFood,
      portions: portions,
      date: new Date().toISOString(),
    };
    setMeals(prevMeals => [newMeal, ...prevMeals]);
    setView('home');
    setDetectedFood(null);
    setSelectedImage(null);
  };

  const handleDeleteMeal = (mealId: string) => {
    setMeals(prevMeals => prevMeals.filter(meal => meal.id !== mealId));
  };

  const handleExportCSV = () => {
    const headers = "Date,Food Name,Portions,Portion Size,Calories (total),Protein (g) (total),Carbs (g) (total),Fat (g) (total)";
    const rows = meals.map(meal => {
        const date = new Date(meal.date).toLocaleString();
        const { name, portionSize, nutrition } = meal.food;
        const { portions } = meal;
        const totalCals = (nutrition.calories * portions).toFixed(0);
        const totalProt = (nutrition.protein * portions).toFixed(1);
        const totalCarbs = (nutrition.carbs * portions).toFixed(1);
        const totalFat = (nutrition.fat * portions).toFixed(1);
        return `"${date}","${name}",${portions},"${portionSize}",${totalCals},${totalProt},${totalCarbs},${totalFat}`;
    });
    
    const csvContent = "data:text/csv;charset=utf-8," + [headers, ...rows].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "nutritrack_history.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  
  const getTodayMeals = () => {
    const today = new Date().toISOString().split('T')[0];
    return meals.filter(meal => meal.date.startsWith(today));
  };

  const todayCalories = getTodayMeals().reduce((total, meal) => {
    return total + meal.food.nutrition.calories * meal.portions;
  }, 0);

  const renderHome = () => (
    <div className="p-4 flex flex-col h-full">
      <header className="text-center mb-6 relative">
        <h1 className="text-4xl font-bold text-slate-800 dark:text-slate-100">NutriTrack</h1>
        <p className="text-slate-500 dark:text-slate-400">Your Argentine Diet Companion</p>
        <button onClick={toggleTheme} className="absolute top-0 right-0 p-2 rounded-full text-slate-500 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700">
            {theme === 'light' ? <MoonIcon className="w-6 h-6" /> : <SunIcon className="w-6 h-6 text-yellow-300" />}
        </button>
      </header>
      
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-slate-700 dark:text-slate-300 mb-2">Today's Progress</h2>
        <div className="relative pt-1">
          <div className="overflow-hidden h-4 mb-2 text-xs flex rounded bg-sky-200 dark:bg-sky-900">
            <div style={{ width: `${Math.min((todayCalories / DAILY_CALORIE_GOAL) * 100, 100)}%` }} className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-sky-500 transition-all duration-500"></div>
          </div>
          <p className="text-right font-semibold text-sky-600 dark:text-sky-400">{todayCalories.toFixed(0)} / {DAILY_CALORIE_GOAL} kcal</p>
        </div>
      </div>
      
      <div className="mb-6 p-4 bg-amber-100 border-l-4 border-amber-400 text-amber-800 rounded-r-lg dark:bg-amber-900/50 dark:border-amber-500 dark:text-amber-300">
        <h3 className="font-bold">Daily Tip</h3>
        <p className="italic">"{dailyTip || 'Loading tip...'}"</p>
      </div>

      <div>
        <h2 className="text-xl font-semibold text-slate-700 dark:text-slate-300 mb-2">Today's Meals</h2>
        <div className="space-y-2">
          {getTodayMeals().length > 0 ? getTodayMeals().map(meal => (
            <div key={meal.id} className="bg-white dark:bg-slate-800 p-3 rounded-lg shadow flex justify-between items-center">
              <div>
                <p className="font-bold dark:text-slate-100">{meal.food.name}</p>
                <p className="text-sm text-slate-500 dark:text-slate-400">{meal.portions} x {meal.food.portionSize} &bull; {(meal.food.nutrition.calories * meal.portions).toFixed(0)} kcal</p>
              </div>
              <button onClick={() => handleDeleteMeal(meal.id)} className="text-red-500 hover:text-red-700 dark:hover:text-red-400 p-1 rounded-full hover:bg-red-100 dark:hover:bg-red-900/50">
                <TrashIcon className="w-5 h-5" />
              </button>
            </div>
          )) : (
            <p className="text-slate-500 dark:text-slate-400 text-center py-4">No meals logged yet today. Tap the camera to start!</p>
          )}
        </div>
      </div>
    </div>
  );

  const renderAnalysis = () => (
    <div className="p-4">
      <div className="flex items-center mb-4">
        <button onClick={() => setView('home')} className="p-2 rounded-full hover:bg-slate-200 dark:hover:bg-slate-700">
          <ChevronLeftIcon className="w-6 h-6" />
        </button>
        <h2 className="text-2xl font-bold text-slate-800 dark:text-slate-100 ml-2">Log Your Meal</h2>
      </div>
      
      {isLoading && (
        <div className="flex flex-col items-center justify-center h-64">
          <Spinner />
          <p className="mt-4 text-slate-600 dark:text-slate-400">Analyzing your delicious meal...</p>
        </div>
      )}

      {error && (
        <div className="text-center p-4 bg-red-100 text-red-700 rounded-lg dark:bg-red-900/50 dark:text-red-300">
          <h3 className="font-bold">Oh no!</h3>
          <p>{error}</p>
        </div>
      )}
      
      {detectedFood && selectedImage && (
        <div className="bg-white dark:bg-slate-800 rounded-lg shadow-lg overflow-hidden">
          <img src={selectedImage} alt={detectedFood.name} className="w-full h-48 object-cover" />
          <div className="p-4">
            <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100">{detectedFood.name}</h3>
            <p className="text-slate-500 dark:text-slate-400 mb-4">{detectedFood.portionSize}</p>
            
            <div className="grid grid-cols-2 gap-4 text-center mb-4">
               <div className="bg-sky-100 dark:bg-sky-900/50 p-2 rounded">
                <p className="font-bold text-sky-800 dark:text-sky-300">{detectedFood.nutrition.calories.toFixed(0)}</p>
                <p className="text-sm text-sky-600 dark:text-sky-400">Calories</p>
              </div>
              <div className="bg-emerald-100 dark:bg-emerald-900/50 p-2 rounded">
                <p className="font-bold text-emerald-800 dark:text-emerald-300">{detectedFood.nutrition.protein.toFixed(1)}g</p>
                <p className="text-sm text-emerald-600 dark:text-emerald-400">Protein</p>
              </div>
              <div className="bg-amber-100 dark:bg-amber-900/50 p-2 rounded">
                <p className="font-bold text-amber-800 dark:text-amber-300">{detectedFood.nutrition.carbs.toFixed(1)}g</p>
                <p className="text-sm text-amber-600 dark:text-amber-400">Carbs</p>
              </div>
              <div className="bg-rose-100 dark:bg-rose-900/50 p-2 rounded">
                <p className="font-bold text-rose-800 dark:text-rose-300">{detectedFood.nutrition.fat.toFixed(1)}g</p>
                <p className="text-sm text-rose-600 dark:text-rose-400">Fat</p>
              </div>
            </div>

            <div className="flex items-center justify-center space-x-4 mb-6">
              <button onClick={() => setPortions(p => Math.max(1, p - 1))} className="p-2 rounded-full bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 disabled:opacity-50" disabled={portions <= 1}>
                <MinusIcon className="w-6 h-6" />
              </button>
              <span className="text-xl font-bold w-12 text-center">{portions}</span>
              <button onClick={() => setPortions(p => p + 1)} className="p-2 rounded-full bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600">
                <PlusIcon className="w-6 h-6" />
              </button>
            </div>
            
            <button
              onClick={handleLogMeal}
              className="w-full bg-sky-500 hover:bg-sky-600 dark:bg-sky-600 dark:hover:bg-sky-700 text-white font-bold py-3 px-4 rounded-lg transition-colors"
            >
              Log {portions} Portion{portions > 1 ? 's' : ''}
            </button>
          </div>
        </div>
      )}
    </div>
  );
  
  const renderHistory = () => {
    const groupedMeals = meals.reduce((acc, meal) => {
        const date = new Date(meal.date).toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
        if (!acc[date]) {
            acc[date] = [];
        }
        acc[date].push(meal);
        return acc;
    }, {} as Record<string, LoggedMeal[]>);

    const today = new Date();
    const lastWeek = new Date(today.getFullYear(), today.getMonth(), today.getDate() - 6);
    lastWeek.setHours(0, 0, 0, 0);

    const weeklyMeals = meals.filter(meal => new Date(meal.date) >= lastWeek);
    const weeklyTotalCalories = weeklyMeals.reduce((total, meal) => total + meal.food.nutrition.calories * meal.portions, 0);

    return (
        <div className="p-4">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-2xl font-bold text-slate-800 dark:text-slate-100">Meal History</h2>
              <button onClick={handleExportCSV} disabled={meals.length === 0} className="flex items-center space-x-2 bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 text-slate-600 dark:text-slate-300 px-3 py-2 rounded-lg text-sm font-semibold disabled:opacity-50 disabled:cursor-not-allowed">
                  <ExportIcon className="w-4 h-4" />
                  <span>Export CSV</span>
              </button>
            </div>

            <div className="mb-6 p-4 bg-white dark:bg-slate-800 rounded-lg shadow">
                <h3 className="font-bold text-lg text-slate-700 dark:text-slate-200">Weekly Summary</h3>
                <p className="text-3xl font-bold text-sky-500 dark:text-sky-400">{weeklyTotalCalories.toFixed(0)}</p>
                <p className="text-sm text-slate-500 dark:text-slate-400">Total calories consumed in the last 7 days.</p>
            </div>

            {Object.keys(groupedMeals).length > 0 ? (
                <div className="space-y-6">
                    {Object.entries(groupedMeals).map(([date, dateMeals]) => (
                        <div key={date}>
                            <h3 className="font-semibold text-slate-600 dark:text-slate-400 mb-2">{date}</h3>
                            <div className="space-y-2">
                                {dateMeals.map(meal => (
                                    <div key={meal.id} className="bg-white dark:bg-slate-800 rounded-lg shadow overflow-hidden">
                                        <div 
                                            className="p-3 flex justify-between items-center cursor-pointer"
                                            onClick={() => setExpandedMealId(prevId => prevId === meal.id ? null : meal.id)}
                                        >
                                            <div>
                                                <p className="font-bold dark:text-slate-100">{meal.food.name}</p>
                                                <p className="text-sm text-slate-500 dark:text-slate-400">{meal.portions} x {meal.food.portionSize} &bull; {(meal.food.nutrition.calories * meal.portions).toFixed(0)} kcal</p>
                                            </div>
                                            <div className="flex items-center space-x-2">
                                              <button 
                                                onClick={(e) => { e.stopPropagation(); handleDeleteMeal(meal.id); }} 
                                                className="text-red-500 hover:text-red-700 dark:hover:text-red-400 p-1 rounded-full hover:bg-red-100 dark:hover:bg-red-900/50"
                                              >
                                                  <TrashIcon className="w-5 h-5" />
                                              </button>
                                              <ChevronLeftIcon className={`w-5 h-5 text-slate-400 transition-transform transform ${expandedMealId === meal.id ? '-rotate-90' : 'rotate-0'}`} />
                                            </div>
                                        </div>
                                        <div className={`transition-all duration-300 ease-in-out ${expandedMealId === meal.id ? 'max-h-40' : 'max-h-0'}`}>
                                            <div className="px-3 pb-3">
                                                <div className="pt-3 border-t border-slate-200 dark:border-slate-700 grid grid-cols-3 gap-2 text-center">
                                                    <div className="bg-emerald-100 dark:bg-emerald-900/50 p-2 rounded">
                                                        <p className="font-bold text-emerald-800 dark:text-emerald-300">{(meal.food.nutrition.protein * meal.portions).toFixed(1)}g</p>
                                                        <p className="text-xs text-emerald-600 dark:text-emerald-400">Protein</p>
                                                    </div>
                                                    <div className="bg-amber-100 dark:bg-amber-900/50 p-2 rounded">
                                                        <p className="font-bold text-amber-800 dark:text-amber-300">{(meal.food.nutrition.carbs * meal.portions).toFixed(1)}g</p>
                                                        <p className="text-xs text-amber-600 dark:text-amber-400">Carbs</p>
                                                    </div>
                                                    <div className="bg-rose-100 dark:bg-rose-900/50 p-2 rounded">
                                                        <p className="font-bold text-rose-800 dark:text-rose-300">{(meal.food.nutrition.fat * meal.portions).toFixed(1)}g</p>
                                                        <p className="text-xs text-rose-600 dark:text-rose-400">Fat</p>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            ) : (
                <p className="text-slate-500 dark:text-slate-400 text-center py-8">No meals have been logged yet.</p>
            )}
        </div>
    );
  };
  
  const renderContent = () => {
    switch(view) {
      case 'home':
        return renderHome();
      case 'analysis':
        return renderAnalysis();
      case 'history':
        return renderHistory();
      default:
        return renderHome();
    }
  };

  return (
    <div className="bg-slate-50 dark:bg-slate-900 min-h-screen font-sans text-slate-900 dark:text-slate-200 flex flex-col">
      <main className="flex-grow pb-20">
        {renderContent()}
      </main>

      <input
        type="file"
        accept="image/*"
        capture="environment"
        ref={fileInputRef}
        onChange={handleImageUpload}
        className="hidden"
      />

      <footer className="fixed bottom-0 left-0 right-0 bg-white dark:bg-slate-800 border-t border-slate-200 dark:border-slate-700 shadow-md">
        <nav className="flex justify-around items-center h-16">
          <button onClick={() => setView('home')} className={`flex flex-col items-center justify-center w-full transition-colors ${view === 'home' ? 'text-sky-500 dark:text-sky-400' : 'text-slate-500 dark:text-slate-400 hover:text-sky-500 dark:hover:text-sky-400'}`}>
            <HomeIcon className="w-7 h-7 mb-1" />
            <span className="text-xs">Home</span>
          </button>
          
          <button onClick={() => fileInputRef.current?.click()} className="transform -translate-y-1/2 bg-sky-500 text-white rounded-full p-4 shadow-lg hover:bg-sky-600 dark:bg-sky-600 dark:hover:bg-sky-500 transition-transform hover:scale-110">
            <CameraIcon className="w-8 h-8" />
          </button>
          
          <button onClick={() => setView('history')} className={`flex flex-col items-center justify-center w-full transition-colors ${view === 'history' ? 'text-sky-500 dark:text-sky-400' : 'text-slate-500 dark:text-slate-400 hover:text-sky-500 dark:hover:text-sky-400'}`}>
            <HistoryIcon className="w-7 h-7 mb-1" />
            <span className="text-xs">History</span>
          </button>
        </nav>
      </footer>
    </div>
  );
};

export default App;