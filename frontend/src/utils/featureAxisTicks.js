const READING_SPELLING_DIFFICULTY_FEATURE =
  "Learning Difficulty – Reading and Spelling Difficulties";

export const getReadingSpellingDifficultyAxisTicks = (
  featureName,
  xValues = [],
  isCategorical = false,
) => {
  if (featureName !== READING_SPELLING_DIFFICULTY_FEATURE) return null;

  return {
    tickmode: "array",
    tickvals: isCategorical
      ? [0, 1].map(
          (value) =>
            xValues.find((xValue) => Number(xValue) === value) ?? String(value),
        )
      : [0, 1],
    ticktext: ["No", "Yes"],
  };
};
