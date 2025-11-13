import { GoogleGenAI, Type } from "@google/genai";
import { DetectedFood } from '../types';

// 👇 Acá es el único cambio importante
const ai = new GoogleGenAI({
  apiKey: import.meta.env.VITE_API_KEY as string,
});

const model = 'gemini-2.5-flash';

const nutritionSchema = {
  type: Type.OBJECT,
  properties: {
    calories: { type: Type.NUMBER, description: 'Estimated calories for the portion.' },
    protein: { type: Type.NUMBER, description: 'Estimated grams of protein.' },
    carbs: { type: Type.NUMBER, description: 'Estimated grams of carbohydrates.' },
    fat: { type: Type.NUMBER, description: 'Estimated grams of fat.' },
  },
  required: ['calories', 'protein', 'carbs', 'fat'],
};

const foodDetectionSchema = {
  type: Type.OBJECT,
  properties: {
    error: { type: Type.STRING, description: 'An error message if the food is not recognized.' },
    name: { type: Type.STRING, description: 'The name of the detected Argentine dish.' },
    portionSize: { type: Type.STRING, description: 'A typical serving size, e.g., "1 unit" or "100g".' },
    nutrition: nutritionSchema,
  },
};

export const estimateNutritionFromImage = async (base64Image: string): Promise<DetectedFood> => {
  const prompt = `You are a nutritional expert specializing in Argentine cuisine. Analyze the attached image. Your task is to identify if the primary food item is one of the following common Argentine dishes: Empanada, Milanesa, Asado, Pizza, Medialuna, Choripán, Locro.

If it is one of these dishes, provide your best estimation for its nutritional values for a standard single serving.

If the food is not one of these, or if you cannot identify it clearly, your response should indicate an error.

Strictly adhere to the JSON schema provided.`;

  try {
    const response = await ai.models.generateContent({
      model,
      contents: {
        parts: [
          {
            inlineData: {
              mimeType: 'image/jpeg',
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
      throw new Error("Could not identify the food. Please try another photo.");
    }

    return result as DetectedFood;
  } catch (error) {
    console.error("Gemini API error:", error);
    throw new Error("Failed to analyze image. The food might not be recognized or there was a network issue.");
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

