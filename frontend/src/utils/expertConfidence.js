export const buildExpertConfidenceBars = (artifact) => {
  const ratingsByExpert = new Map();

  for (const feature of artifact?.submissions_by_feature || []) {
    for (const submission of feature?.submissions || []) {
      const userName = String(submission?.user_name || "").trim();
      const rating = Number(submission?.sureness);
      if (
        !userName ||
        submission?.sureness == null ||
        !Number.isFinite(rating) ||
        rating < 1 ||
        rating > 10
      ) {
        continue;
      }

      const totals = ratingsByExpert.get(userName) || { sum: 0, count: 0 };
      totals.sum += rating;
      totals.count += 1;
      ratingsByExpert.set(userName, totals);
    }
  }

  return [...ratingsByExpert]
    .map(([userName, { sum, count }]) => ({
      userName,
      averageConfidence: sum / count,
      ratingCount: count,
    }))
    .sort(
      (left, right) =>
        right.averageConfidence - left.averageConfidence ||
        left.userName.localeCompare(right.userName),
    );
};
