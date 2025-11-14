import { GoogleGenAI, Type } from "@google/genai";
import { DetectedFood } from "../types";

const ai = new GoogleGenAI({
  apiKey: import.meta.env.VITE_API_KEY as string,
});

const model = "gemini-2.5-flash";

const nutritionSchema = {
  type: Type.OBJECT,
  properties: {
    calories: {
      type: Type.NUMBER,
      description: "Estimated calories for the portion.",
    },
    protein: {
      type: Type.NUMBER,
      description: "Estimated grams of protein.",
    },
    carbs: {
      type: Type.NUMBER,
      description: "Estimated grams of carbohydrates.",
    },
    fat: {
      type: Type.NUMBER,
      description: "Estimated grams of fat.",
    },
  },
  required: ["calories", "protein", "carbs", "fat"],
};

const foodDetectionSchema = {
  type: Type.OBJECT,
  properties: {
    error: {
      type: Type.STRING,
      description:
        "An error message if the food is not recognized or the image has no food.",
    },
    name: {
      type: Type.STRING,
      description: "The name of the detected dish or food.",
    },
    portionSize: {
      type: Type.STRING,
      description:
        'A typical serving size, e.g., "1 unit", "1 plate", "1 slice", "1 bowl", "100g".',
    },
    nutrition: nutritionSchema,
  },
};

export const estimateNutritionFromImage = async (
  base64Image: string
): Promise<DetectedFood> => {
  const prompt = `
You are a nutritional expert specialized in Argentine food and everyday international dishes.
Analyze the attached image and identify the SINGLE primary food or dish.

Your goals:
- Always choose the closest reasonable dish/food, even if it is not a perfect match.
- Only use "error" if the image clearly does not contain food or is impossible to interpret.

### A. Argentine main dishes (prefer these when they match)
- Empanada de carne
- Empanada de pollo
- Empanada de jamón y queso
- Milanesa de carne
- Milanesa de pollo
- Milanesa napolitana
- Asado (carne vacuna a la parrilla)
- Chorizo a la parrilla / Choripán
- Provoleta
- Pollo al horno con papas
- Pastel de papa
- Locro
- Humita en chala
- Pizza muzzarella
- Pizza napolitana
- Pizza de fugazzeta
- Pizzanesa
- Fideos con salsa de tomate (fideos con tuco)
- Fideos con manteca y queso rallado
- Ñoquis de papa con salsa
- Arroz con pollo
- Polenta con salsa de tomate y queso
- Hamburguesa casera con pan
- Sandwich de milanesa
- Ensalada mixta (lechuga, tomate, cebolla)
- Ensalada rusa (papa, zanahoria, arvejas, mayonesa)
- Tarta de verdura o jamón y queso
- Tortilla de papa (tortilla española)
- Pancho (hot dog argentino)

### B. Common international / generic dishes
- Pasta con salsa de tomate (spaghetti, penne, etc.)
- Pasta con salsa blanca / crema
- Lasagna
- Pizza de pepperoni
- Cheeseburger (hamburguesa con queso)
- Hot dog
- Sushi (rolls variados)
- Tacos de carne
- Tacos de pollo
- Burrito
- Wrap de pollo
- Curry de pollo con arroz
- Bowl de arroz con vegetales
- Pollo grillado con guarnición
- Filete de salmón grillado
- Fish and chips (pescado frito con papas)
- Stir-fry de vegetales con arroz
- Sandwich de jamón y queso
- Omelette con queso y vegetales
- Sopa (sopa de verduras, sopa de pollo, sopa cremosa)

### C. Fruits (as main item)
- Banana
- Manzana
- Naranja
- Mandarina
- Pera
- Uvas
- Sandía
- Melón
- Frutilla / fresa
- Arándanos
- Kiwi
- Ananá / piña
- Durazno
- Ciruela

### D. Vegetables / sides
- Papa hervida
- Papa al horno
- Papa frita (french fries)
- Batata / camote
- Zanahoria
- Tomate
- Lechuga
- Cebolla
- Morrón / pimiento
- Zucchini / zapallito
- Brócoli
- Espinaca
- Repollo
- Mix de vegetales salteados
- Ensalada variada en bowl

### E. Desserts and sweets
- Helado en pote (1 o 2 bochas)
- Flan con dulce de leche
- Panqueque con dulce de leche
- Brownie
- Torta de chocolate
- Tiramisu
- Alfajor de dulce de leche
- Factura (medialuna, vigilante, etc.)
- Galletitas dulces

### F. Drinks (if they are clearly the main item)
- Vaso de agua
- Gaseosa (cola)
- Jugo de fruta
- Café
- Mate
- Cerveza

### Behavior rules

1. ALWAYS try to classify the food as one of the items above or the closest similar dish.
   - Example: any spaghetti-like pasta with red sauce -> "Pasta con salsa de tomate".
   - Example: any grilled steak with side -> "Asado" or "meat dish" depending on the image.
2. If there are multiple items on the plate, choose the one that visually occupies the most space.
3. For portionSize, use natural labels like:
   - "1 unit", "2 units"
   - "1 plate", "1 slice", "1 bowl"
   - "100g approx."
4. Nutrition values must be your best realistic estimate for that portion:
   - calories (kcal)
   - protein (grams)
   - carbs (grams)
   - fat (grams)
5. ONLY use error if:
   - The image clearly has no food (for example, a car, a person, a wall), or
   - It is so blurred or dark that you cannot even guess the type of food.

You MUST return a JSON object that matches this structure:

{
  "error": string | null,
  "name": string,
  "portionSize": string,
  "nutrition": {
    "calories": number,
    "protein": number,
    "carbs": number,
    "fat": number
  }
}
`;

  try {
    const response = await ai.models.generateContent({
      model,
      contents: {
        parts: [
          {
            inlineData: {
              mimeType: "image/jpeg",
              data: base64Image,
            },
          },
          { text: prompt },
        ],
      },
      config: {
        responseMimeType: "application/json",
        responseSchema: foodDetectionSchema,
      },
    });

    const jsonString = response.text.trim();
    const result = JSON.parse(jsonString);

    if (result.error) {
      throw new Error(result.error);
    }

    if (!result.name || !result.nutrition) {
      throw new Error(
        "Could not identify the food. Please try another photo."
      );
    }

    return result as DetectedFood;
  } catch (error) {
    console.error("Gemini API error:", error);
    throw new Error(
      "Failed to analyze image. The food might not be recognized or there was a network issue."
    );
  }
};

export const getDailyTip = async (): Promise<string> => {
  const prompt =
    "Generate a single, concise, and encouraging nutritional tip for the day, relevant to Argentine culture. Keep it under 25 words. For example: 'Enjoy a glass of Malbec, but in moderation!' or 'A walk after asado aids digestion.'";

  try {
    const response = await ai.models.generateContent({
      model,
      contents: prompt,
    });
    return response.text;
  } catch (error) {
    console.error("Gemini tip generation error:", error);
    return "Stay hydrated by drinking plenty of water throughout the day!";
  }
};
