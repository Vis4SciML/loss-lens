import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function roundToPercentage(number: number | string) {
  if (typeof number === "string") {
    number = parseFloat(number)
  }
  // Multiply the number by 100 to convert it to a percentage
  const percentage = number * 100

  // Round the percentage to two decimal places
  const roundedPercentage = Math.round(percentage * 100) / 100

  // Add the percentage symbol ("%") to the rounded percentage
  return roundedPercentage + "%"
}
