export const LANGUAGE_STORAGE_KEY = "gamweaver-language";
export const DEFAULT_LANGUAGE = "en";

const translations = {
  en: {
    "language.switchToEnglish": "Switch language to English",
    "language.switchToGerman": "Switch language to German",

    "common.yes": "Yes",
    "common.no": "No",
    "common.cancel": "Cancel",
    "common.username": "Username",
    "common.password": "Password",
    "common.profession": "Profession",
    "common.loading": "Loading...",
    "common.refresh": "Refresh",
    "common.save": "Save",
    "common.apply": "Apply",
    "common.reset": "Reset",
    "common.close": "Close",
    "common.delete": "Delete",
    "common.target": "Target",
    "common.unknown": "Unknown",

    "header.subtitle": "Interactive GAM Editor",
    "header.modelTrained": "Model Trained",
    "header.modelNotTrained": "Model Not Trained",
    "header.recordsLoaded": "{count} records loaded",

    "login.title": "The Interactive GAM Editor",
    "login.subtitleLogin": "Sign in with your credentials",
    "login.subtitleRegister": "Register with an invite link",
    "login.usernamePlaceholder": "Your username",
    "login.passwordPlaceholder": "Your password",
    "login.professionPlaceholder": "Your profession",
    "login.registerInviteOnly": "Registration is only possible via an invite link.",
    "login.error.enterCredentials": "Please enter username and password",
    "login.error.professionRequired": "Please enter your profession",
    "login.error.inviteRequired": "Registration requires a valid invite link",
    "login.error.loginFailed": "Failed to login",
    "login.error.registerFailed": "Failed to register",
    "login.error.serverUnavailable":
      "Unable to reach the server. Please try again.",
    "login.submitSigningIn": "Signing in...",
    "login.submitRegistering": "Registering...",
    "login.submitSignIn": "Sign In",
    "login.submitRegister": "Register",

    "app.loggedInAs": "Logged in as",
    "app.viewCombinedResults": "View Combined Results",
    "app.superadmin": "Superadmin",
    "app.logout": "Logout",
    "app.footerDescription":
      "GAMWeaver - Interactive GAM Editor for interactive modeling and explainability. Developed by {name} as part of a master's thesis.",
    "app.githubLink": "GAMWeaver GitHub",
    "app.notification.editRemoved": "Edit Removed",
    "app.notification.editsRemoved": "Edits Removed",
    "app.notification.singleMessage":
      "One of your edits was removed by another user:",
    "app.notification.multipleMessage":
      "{count} of your edits were removed by other users:",
    "app.notification.feature": "Feature",
    "app.notification.by": "by",
    "app.notification.curveEdit": "Curve edit",
    "app.notification.pointCount": "Points",
    "app.notification.xSummary": "X Summary",
    "app.notification.xValue": "X Value",
    "app.notification.reason": "Reason",
    "app.notification.gotIt": "Got it",
    "app.error.applyShapeEdits": "Failed to apply shape function edits",
    "app.error.resetShapeFunctions": "Failed to reset shape functions",
    "app.error.resetFeature": "Failed to reset feature",
    "app.error.updateFeatureChartSettings":
      "Failed to update feature chart settings",
    "app.error.loadData": "Failed to load data",
    "app.error.uploadDataset": "Failed to upload dataset",
    "app.error.trainModel": "Failed to train model",
    "app.error.predictionFailed": "Prediction failed",

    "training.title": "Model Training",
    "training.restrictedNotice":
      "Dataset upload/loading and training controls are restricted to the superadmin.",
    "training.dataLoaded": "Data loaded",
    "training.modelTrainedLabel": "Model trained",
    "training.activeDataset": "Active dataset",
    "training.stepLoadData": "Load Data",
    "training.stepLoaded": "Loaded",
    "training.loadDescription":
      "Upload a CSV dataset, select the prediction target, and choose which columns to import.",
    "training.uploadAndLoad": "Upload & Load Dataset",
    "training.stepTrainModel": "Train Model",
    "training.stepTrained": "Trained",
    "training.trainDescription": "Configure and train the IGANN model.",
    "training.estimators": "Number of Estimators",
    "training.estimatorHint":
      "More estimators = better accuracy but slower training",
    "training.trainButton": "Train IGANN Model",
    "training.uploadModalTitle": "Upload Dataset",
    "training.csvFile": "CSV File",
    "training.useCaseStudyFeatureNames":
      "Use Case study specific feature names",
    "training.uploadInspect": "Upload & Inspect Columns",
    "training.columnsDetected": "Columns detected",
    "training.targetColumn": "Target Column",
    "training.featureColumns": "Feature Columns",
    "training.expandFeatureColumns": "Expand feature columns",
    "training.collapseFeatureColumns": "Collapse feature columns",
    "training.selectAll": "Select all",
    "training.clearAll": "Clear all",
    "training.target": "Target",
    "training.selectedFeatures": "Selected features",
    "training.loadDataset": "Load Dataset",
    "training.error.chooseCsvFirst": "Please choose a CSV file first.",
    "training.error.uploadFailed": "Failed to upload dataset",
    "training.error.loadFailed": "Failed to load dataset",
    "training.error.uploadNeedsSelections":
      "Please upload a CSV, choose a target column, and select at least one feature column.",
    "training.progress.loadingData": "Loading data...",
    "training.progress.training":
      "Training model... This may take a moment.",
    "training.uploading": "Uploading...",
    "training.loadingDataset": "Loading...",
    "training.training": "Training...",
    "training.statusDataLoaded": "Data loaded",
    "training.statusModelTrained": "Model trained",
    "training.statusModelSource": "Model source",
    "training.modelSourceImported": "imported",
    "training.modelSourceTrained": "trained",
    "training.importedAnalyticsUnavailable":
      "Imported model analytics stay unavailable until the superadmin loads a compatible dataset.",

    "dataSummary.title": "Data Summary",
    "dataSummary.empty": "Load the data to see summary statistics",
    "dataSummary.totalRecords": "Total Records",
    "dataSummary.targetVariable": "Target Variable",
    "dataSummary.targetFallback": "Target",
    "dataSummary.mean": "Mean",
    "dataSummary.stdDev": "Std Dev",
    "dataSummary.min": "Min",
    "dataSummary.max": "Max",
    "dataSummary.features": "Features",

    "prediction.title": "Make Prediction",
    "prediction.description":
      "Set dataset feature values and run a prediction. Inputs are generated from the active dataset schema.",
    "prediction.trainFirst":
      "Please train the model first before making predictions.",
    "prediction.noSchema":
      "No feature schema available. Ask the superadmin to load a dataset.",
    "prediction.predicting": "Predicting...",
    "prediction.predictTarget": "Predict {target}",
    "prediction.predictedTarget": "Predicted {target}",
    "prediction.scaleHelpLabel": "Explain the value scale for {feature}",
    "prediction.scaleTooltipOptionsIntro":
      "These values use the same x-axis coding:",
    "prediction.scaleTooltipAvailableValues":
      "Available x-axis values: {values}.",
    "prediction.scaleTooltipRange":
      "This input uses the same x-axis scale: {min} to {max}.",
    "prediction.scaleTooltipSingleValue":
      "This input uses a fixed x-axis value of {value}.",
    "prediction.competenceLevel": "Competence Level",
    "prediction.scoreValue": "Score",
    "prediction.assignment": "Assignment",
    "prediction.target.germanReading": "German reading literacy",
    "prediction.option.TR_TLS1.0": "No",
    "prediction.option.TR_TLS1.1": "Yes",
    "prediction.option.KBuecher.1": "0–10 books",
    "prediction.option.KBuecher.2": "11–25 books",
    "prediction.option.KBuecher.3": "26–100 books",
    "prediction.option.KBuecher.4": "101–200 books",
    "prediction.option.KBuecher.5": "201–500 books",
    "prediction.option.KBuecher.6": "More than 500 books",
    "prediction.option.KDezh16_r.1": "Never speaks German at home",
    "prediction.option.KDezh16_r.2": "Speaks German and another language at home",
    "prediction.option.KDezh16_r.3": "Always or almost always speaks German at home",

    "shapeFunctions.title": "Shape Functions",
    "shapeFunctions.emptyDescription":
      "Train the model to see feature shape functions.",
    "shapeFunctions.sharedDescription":
      "Shape functions show how each feature affects the prediction.",
    "shapeFunctions.gridDescription":
      "These plots show how each feature affects the bike rental prediction. Values above zero increase the prediction, values below decrease it.",
    "shapeFunctions.effectOnPrediction": "Effect on Prediction",
    "shapeFunctions.interactiveTitle": "Interactive Shape Functions",
    "shapeFunctions.interactiveEmptyDescription":
      "Train the model to see and edit feature shape functions.",
    "shapeFunctions.interactiveEditDescription":
      "Edit points, then click Submit on each feature to save with your confidence rating.",
    "shapeFunctions.interactiveViewDescription":
      "Enable editing mode to interactively modify shape functions.",
    "shapeFunctions.resetAll": "Reset All",
    "shapeFunctions.syncAxes": "Sync Axes",
    "shapeFunctions.axesSynced": "Axes Synced",
    "shapeFunctions.centerCurves": "Center curves",
    "shapeFunctions.centerCurvesHint":
      "Subtract the mean of each curve for this display only.",
    "shapeFunctions.syncAxesHintOn":
      "Axes are synced - click to use per-chart scale",
    "shapeFunctions.syncAxesHintOff":
      "Click to sync all chart axes to the same scale",
    "shapeFunctions.enableEditing": "Enable Editing",
    "shapeFunctions.editingModeOn": "Editing Mode ON",
    "shapeFunctions.editHelp":
      "For numeric features, click and drag to brush edits along the curve. Use the Soft-Hard slider to control how much of the curve is affected. For categorical features, hover a point, then drag it up or down. Double-click for precise value entry. When done editing a feature, click its Submit button and rate your confidence.",
    "shapeFunctions.lineBrushHardness": "Line Brush Hardness",
    "shapeFunctions.soft": "Soft",
    "shapeFunctions.hard": "Hard",
    "shapeFunctions.original": "Original",
    "shapeFunctions.edited": "Edited",
    "shapeFunctions.current": "Current",
    "shapeFunctions.comparison": "Comparison",
    "shapeFunctions.missing": "Missing",
    "shapeFunctions.editPoints": "Edit points",
    "shapeFunctions.effect": "Effect",
    "shapeFunctions.draggingSuffix": "(dragging...)",
    "shapeFunctions.editedSuffix": "(edited)",
    "shapeFunctions.setEffectValue": "Set Effect Value",
    "shapeFunctions.brushingAtX": "Brushing at x = {value}",
    "shapeFunctions.releaseToApply":
      "Release to apply smoothed stroke: {value}",
    "shapeFunctions.draggingPoint": "Dragging point {index}",
    "shapeFunctions.releaseToSetValue": "Release to set value: {value}",
    "shapeFunctions.clickDragHint":
      "Click and drag to brush the line. Double-click for precise entry.",
    "shapeFunctions.pointSelected": "Point {index} selected",
    "shapeFunctions.preciseValueHint": "Double-click for precise value",
    "shapeFunctions.hoverSelectHint": "Hover over a point to select it",
    "shapeFunctions.submitEditTitle": "Submit Edit for {feature}",
    "shapeFunctions.submitEditDescription":
      "Rate your confidence and provide a description for your edit.",
    "shapeFunctions.confidenceLevel": "Confidence Level",
    "shapeFunctions.notSure": "Not sure",
    "shapeFunctions.verySure": "Very sure",
    "shapeFunctions.editDescription": "Edit Description",
    "shapeFunctions.editDescriptionPlaceholder":
      "Describe why you made this edit...",
    "shapeFunctions.editDescriptionRequired":
      "Please enter a description for your edit.",
    "shapeFunctions.submitEdit": "Submit Edit",
    "shapeFunctions.chartSettingsSaveError":
      "Failed to save chart settings",
    "shapeFunctions.chartMappingTitle": "Chart Mapping: {feature}",
    "shapeFunctions.chartMappingDescription":
      "Configure chart type and x-axis labels for all users.",
    "shapeFunctions.displayTitleLabel": "Display Title",
    "shapeFunctions.displayTitleHint":
      "Cosmetic only. Charts and UI labels use this title, but exports keep the original feature name.",
    "shapeFunctions.xAxisLabel": "X-Axis Label",
    "shapeFunctions.xAxisLabelHint":
      "Optional. Leave blank to use the display title on the x-axis.",
    "shapeFunctions.chartType": "Chart Type",
    "shapeFunctions.numericChartType": "Numeric",
    "shapeFunctions.categoricalChartType": "Categorical",
    "shapeFunctions.numericMappingHint":
      "Assign display labels to whole-number numeric x-axis values while keeping the axis continuous.",
    "shapeFunctions.categoricalMappingHint":
      "Assign display labels to categorical x-axis values shown on this chart.",
    "shapeFunctions.treatAsCategorical":
      "Treat this chart as categorical",
    "shapeFunctions.treatAsCategoricalHint":
      "Use bars with discrete category labels instead of a continuous line.",
    "shapeFunctions.cannotConvertCategorical":
      "This feature has too many or non-discrete values and cannot be converted to a categorical chart.",
    "shapeFunctions.xAxisValueLabels": "X-Axis Value Labels",
    "shapeFunctions.numericXAxisLabels": "Whole-Number X-Axis Labels",
    "shapeFunctions.noCategoricalValues":
      "No categorical values available for mapping.",
    "shapeFunctions.noNumericValues":
      "No numeric x-axis values available for mapping.",
    "shapeFunctions.displayLabelFor": "Display label for {value}",
    "shapeFunctions.mappingAvailableAfterCategorical":
      "Label mapping is available once this feature is displayed as a categorical chart.",
    "shapeFunctions.saveMapping": "Save Mapping",
    "shapeFunctions.saving": "Saving...",
    "shapeFunctions.switchChartToNumeric": "Switch chart to numeric",
    "shapeFunctions.switchChartToCategorical":
      "Switch chart to categorical",
    "shapeFunctions.chartTypeUnavailable":
      "This feature cannot be switched to the other chart type",
    "shapeFunctions.editCategoricalMapping":
      "Edit categorical mapping",
    "shapeFunctions.editChartMapping":
      "Edit chart mapping",
    "shapeFunctions.chartMappingButton": "Chart Mapping",
    "shapeFunctions.enlargeChart": "Enlarge chart",
    "shapeFunctions.submitFeature": "Submit",
    "shapeFunctions.dataDistribution": "Data Distribution",
    "shapeFunctions.showDistribution": "Show distribution",
    "shapeFunctions.hideDistribution": "Hide distribution",
    "shapeFunctions.countAxisLabel": "Count",
    "shapeFunctions.distributionTotalCount":
      "Non-missing datapoints used: {count}",
    "shapeFunctions.distributionAnsweredCount":
      "Answered / non-missing: {count}",
    "shapeFunctions.distributionMissingCount": "Missing: {count}",
    "shapeFunctions.distributionRepresentedCount":
      "Total rows represented: {count}",
    "shapeFunctions.distributionEmpty": "No distribution data available.",
    "shapeFunctions.openFeatureDetails": "Open feature details",
    "shapeFunctions.detailsButton": "Details",
    "shapeFunctions.detailsDrawerTitle": "Feature Details",
    "shapeFunctions.detailsSummary":
      "Shows how this feature was constructed in the dataset.",
    "shapeFunctions.detailsUnavailable":
      "No feature provenance metadata is available for this feature.",
    "shapeFunctions.detailsCategory": "Category",
    "shapeFunctions.detailsConstruction": "Construction",
    "shapeFunctions.detailsSourceVariables": "Source variables",
    "shapeFunctions.detailsSourceQuestions": "Source Variables",
    "shapeFunctions.detailsAnswerScale": "Answer scale",
    "shapeFunctions.detailsScaleItems": "Source questions",
    "shapeFunctions.detailsScaleItemsSummary":
      "This is an imputed scale score derived from the following statements.",
    "shapeFunctions.detailsTransformation": "Transformation",
    "shapeFunctions.detailsMissingHandling": "Missing values",
    "shapeFunctions.detailsRationale": "Why this feature was selected",
    "shapeFunctions.detailsSourceCount": "{count} source questions",
    "shapeFunctions.detailsRawSource": "Raw source value",
    "shapeFunctions.detailsItemMean": "Item mean",
    "shapeFunctions.detailsIqbScale": "IQB scale",
    "shapeFunctions.detailsUnknownConstruction": "Derived feature",
    "shapeFunctions.detailAnswerOption.agreement.1": "Strongly disagree",
    "shapeFunctions.detailAnswerOption.agreement.2": "Rather disagree",
    "shapeFunctions.detailAnswerOption.agreement.3": "Rather agree",
    "shapeFunctions.detailAnswerOption.agreement.4": "Strongly agree",
    "shapeFunctions.detailQuestion.Sdruckd01a":
      "In German class, a lot is expected of us.",
    "shapeFunctions.detailQuestion.Sdruckd01b":
      "We often have unannounced tests in German class.",
    "shapeFunctions.detailQuestion.Sdruckd01c":
      "If you are sick too often, you quickly fall behind in German class.",
    "shapeFunctions.detailQuestion.Sdruckd01d":
      "If you do not study at the weekend, you can hardly keep up in German class.",
    "shapeFunctions.detailQuestion.Sdruckd01e":
      "We can hardly keep up with the homework in German class.",
    "shapeFunctions.detailQuestion.Ssoe01a": "I am popular in my class.",
    "shapeFunctions.detailQuestion.Ssoe01b":
      "The other children in my class come to me with their problems.",
    "shapeFunctions.detailQuestion.Ssoe01c":
      "I have really good friends in my class.",
    "shapeFunctions.detailQuestion.Ssoe01d":
      "The other children in my class like me.",
    "shapeFunctions.detailQuestion.Sind01a": "German is important.",
    "shapeFunctions.detailQuestion.Sind01b": "German is fun.",
    "shapeFunctions.detailQuestion.Sind01c": "I am interested in German.",
    "shapeFunctions.detailQuestion.Sind01d":
      "German is one of my favourite subjects.",
    "shapeFunctions.detailQuestion.Sword01a":
      "I worry that there will not be enough time in a German test.",
    "shapeFunctions.detailQuestion.Sword01b":
      "I worry that my teacher will call on me in German class.",
    "shapeFunctions.detailQuestion.Sword01c":
      "I worry that I cannot solve my German homework.",
    "shapeFunctions.detailQuestion.Sword01d":
      "I worry that I will not understand something in German class.",
    "shapeFunctions.detailQuestion.Semod01a":
      "I get nervous when there is not enough time in a German test.",
    "shapeFunctions.detailQuestion.Semod01b":
      "I get nervous when my teacher calls on me in German class.",
    "shapeFunctions.detailQuestion.Semod01c":
      "I get nervous when I cannot solve my German homework.",
    "shapeFunctions.detailQuestion.Semod01d":
      "I get nervous when I do not understand something in German class.",
    "shapeFunctions.detailQuestion.Sumz21a": "I can type using a keyboard.",
    "shapeFunctions.detailQuestion.Sumz21b": "I can click using a mouse.",
    "shapeFunctions.detailQuestion.Sumz21c":
      "I can work on assignments on a tablet.",
    "shapeFunctions.detailQuestion.Sumz21d":
      "I can work on assignments on a computer.",
    "shapeFunctions.detailQuestion.Sumz21e":
      "I can find something using a search engine.",
    "shapeFunctions.detailQuestion.Sulkd01a": "My German teacher cares about me.",
    "shapeFunctions.detailQuestion.Sulkd01b": "My German teacher encourages me.",
    "shapeFunctions.detailQuestion.Sulkd01c": "My German teacher gives me advice.",
    "shapeFunctions.detailQuestion.Sulkd01d":
      "My German teacher tells me what I am already good at and what I still need to learn.",
    "shapeFunctions.detailQuestion.Sulkd01e":
      "My German teacher takes time for me.",
    "shapeFunctions.detailQuestion.Sulkd01f": "My German teacher praises me.",
    "shapeFunctions.detailQuestion.Sulkd01g":
      "My German teacher gives me suggestions for improvement.",

    "predictionComparison.title": "Prediction Comparison",
    "predictionComparison.empty":
      "Edit shape functions and apply changes to see prediction comparison.",
    "predictionComparison.subtitleWithEdits":
      "Compare predictions between original IGANN model and your edited version.",
    "predictionComparison.subtitleWithoutEdits":
      "Edit shape functions to see how predictions change.",
    "predictionComparison.customizeColors": "Customise chart colours",
    "predictionComparison.colorsButton": "Colours",
    "predictionComparison.chartColors": "Chart Colours",
    "predictionComparison.originalModelDots": "Original model dots",
    "predictionComparison.editedModelDots": "Edited model dots",
    "predictionComparison.perfectPredictionLine":
      "Perfect prediction line",
    "predictionComparison.persistHint":
      "Log in to persist colours across sessions.",
    "predictionComparison.originalTrace": "IGANN (Original)",
    "predictionComparison.editedTrace":
      "IGANN Interactive (Edited)",
    "predictionComparison.perfectPrediction": "Perfect Prediction",
    "predictionComparison.plotTitle": "Predictions vs Actual",
    "predictionComparison.plotTitleWithEdits":
      "Predictions vs Actual (Original vs Edited)",
    "predictionComparison.actualBikeRentals": "Actual Bike Rentals",
    "predictionComparison.predictedBikeRentals":
      "Predicted Bike Rentals",
    "predictionComparison.originalModel": "IGANN (Original)",
    "predictionComparison.editedModel":
      "IGANN Interactive (Edited)",
    "predictionComparison.summary": "Summary:",
    "predictionComparison.improved": "improved",
    "predictionComparison.worsened": "worsened",

    "combined.title": "Combined Results - All Users",
    "combined.refresh": "Refresh",
    "combined.loading": "Loading combined results...",
    "combined.summaryTitle": "Combined Analysis from {count} Users",
    "combined.summaryDescription":
      "This page shows the aggregated effect of all user edits on the GAM model. Each point's offset is averaged across all users who edited it.",
    "combined.participatingUsers": "Participating Users ({count})",
    "combined.noUsers": "No users yet.",
    "combined.originalModel": "Original Model",
    "combined.combinedUserEdits": "Combined User Edits",
    "combined.rmse": "RMSE",
    "combined.mae": "MAE",
    "combined.r2": "R2",
    "combined.scatterTitle": "Predicted vs Original (Scatter Plot)",
    "combined.scatterDescription":
      "Points closer to the diagonal line indicate better predictions. Compare how the combined user edits affect prediction accuracy.",
    "combined.formTitle": "Predict with Combined Edits",
    "combined.formDescription":
      "Make a prediction using the model with all combined user edits applied. The result reflects the aggregated shape function modifications from all users.",
    "combined.predictButton": "Predict {target} (Combined Model)",
    "combined.predictedResult":
      "Predicted {target} (Combined Edits)",
    "combined.predictionFailed": "Prediction failed",
    "combined.shapeFunctionsNone": "No shape function data available.",
    "combined.shapeFunctionsTitle":
      "Shape Functions: Original vs Combined Edits",
    "combined.weightingToggleTitle":
      "Toggle whether the combined line uses confidence-weighted averaging or a simple mean",
    "combined.weightingOn": "Weighting: On",
    "combined.weightingOff": "Weighting: Off",
    "combined.showUserOverlay": "Show User Overlay",
    "combined.loadOverlayFailed": "Could not load user overlays. Please try again.",
    "combined.updateWeightingFailed": "Could not change weighting. Please try again.",
    "combined.hideUserOverlay": "Hide User Overlay",
    "combined.legend": "Legend:",
    "combined.original": "Original",
    "combined.combined": "Combined",
    "combined.comparison": "Comparison Dataset",
    "combined.noEdits":
      "No user edits have been made yet. The charts below show the original shape functions.",
    "combined.modified": "Modified",
    "combined.effect": "Effect",
    "combined.editLogsUnavailable":
      "Edit logs are temporarily unavailable: {error}",
    "combined.editLogsTitle": "Edit Logs",
    "combined.editLogsDescription":
      "Detailed log of all user edits, grouped by feature. Shows who edited, their self-reported confidence rating (1-10), raw input value, and the weighted result applied to the combined view.",
    "combined.submissionSingular": "submission",
    "combined.submissionPlural": "submissions",
    "combined.userSingular": "user",
    "combined.userPlural": "users",
    "combined.by": "by",
    "combined.user": "User",
    "combined.submitted": "Submitted",
    "combined.confidence": "Confidence",
    "combined.points": "Points",
    "combined.rawTotal": "Raw Total",
    "combined.weightedTotal": "Weighted Total",
    "combined.message": "Message",
    "combined.xSummary": "X Summary",
    "combined.deleteSubmission": "Delete Submission",
    "combined.xValue": "X Value",
    "combined.rawInput": "Raw Input",
    "combined.weighted": "Weighted",
    "combined.originalLinePreview": "Original Line",
    "combined.weightedLinePreview": "Weighted Edited Line",
    "combined.noSubmissions":
      "No submissions available for this feature.",
    "combined.deleteDisabled":
      "Destructive actions are disabled in this demo.",
    "combined.deleteReasonRequired":
      "Please provide a reason for deleting this edit.",
    "combined.noDeletionTarget":
      "No deletion target available for this submission",
    "combined.deleteFailed": "Failed to delete submission",
    "combined.loadEditLogsFailed": "Failed to load edit logs",
    "combined.loadDataFailed": "Failed to load data",
    "combined.deleteModalTitle": "Delete Submission",
    "combined.deleteModalDescription":
      "You are about to delete a submitted curve edit by {user}.",
    "combined.feature": "Feature",
    "combined.whyRemoved":
      "Why is this submission being removed?",
    "combined.deleteReasonPlaceholder":
      "Provide a reason for removing this submission...",
    "combined.deleting": "Deleting...",

    "superadmin.error.loadUsers": "Failed to load users",
    "superadmin.error.usernamePasswordRequired":
      "Username and password are required",
    "superadmin.error.usernamePasswordProfessionRequired":
      "Username, password, and profession are required",
    "superadmin.error.createUser": "Failed to create user",
    "superadmin.error.createInvite": "Failed to create invite",
    "superadmin.error.resetDatabase": "Failed to reset database",
    "superadmin.error.exportModel": "Failed to export model",
    "superadmin.error.importModel": "Failed to import model",
    "superadmin.compareUploadError": "Failed to upload comparison dataset",
    "superadmin.compareLoadError": "Failed to load comparison dataset",
    "superadmin.title": "Superadmin Overview",
    "superadmin.openCombined": "Open Combined Results",
    "superadmin.openModelCompare": "Compare Model Exports",
    "superadmin.compareDatasets": "Compare datasets",
    "superadmin.compareDatasetsDescription":
      "Upload a second dataset, train it with the primary estimator count, and overlay its shape functions in the interactive charts.",
    "superadmin.compareModalDescription":
      "The comparison dataset must use the same target column and selected feature columns as the primary dataset.",
    "superadmin.uploadAndSelectComparison":
      "Upload & Select Features",
    "superadmin.trainComparison": "Train Comparison Model",
    "superadmin.primaryEstimators": "Primary estimators",
    "superadmin.comparisonDataset": "Comparison dataset",
    "superadmin.noComparisonDataset": "No comparison dataset loaded",
    "superadmin.comparisonTrained": "Comparison trained",
    "superadmin.loadComparisonDataset": "Load Comparison Dataset",
    "superadmin.sensitivityTitle": "Preprocessing sensitivity check",
    "superadmin.sensitivityDescription":
      "Upload the original full and 5% IQB CSVs with their matching unedited IGANN exports. Keep the original CSV filenames. Refit both models using medians learned only from their own training rows, then evaluate each on its original 20% test split. The active model is unaffected.",
    "superadmin.sensitivityFullCsv": "Full-data CSV",
    "superadmin.sensitivityFullModel": "Original full-data model JSON",
    "superadmin.sensitivityReducedCsv": "5% CSV",
    "superadmin.sensitivityReducedModel": "Original 5% model JSON",
    "superadmin.sensitivityRun": "Run sensitivity check",
    "superadmin.sensitivityRunning": "Retraining both models...",
    "superadmin.sensitivityError": "Sensitivity check failed",
    "superadmin.sensitivityMethodNote":
      "The exported model is scored from its saved shape functions, which may approximate native predictions. The controlled delta compares two native refits with identical settings and test rows; only the median-fitting scope changes. Lower MAE and RMSE are better; higher R² is better.",
    "superadmin.sensitivityFullResult": "Full-data model",
    "superadmin.sensitivityReducedResult": "5% model",
    "superadmin.sensitivityRowCounts":
      "{usable} usable rows · {train} training · {test} test",
    "superadmin.sensitivityMaePercent":
      "MAE change from the controlled original refit: {percent}%",
    "superadmin.sensitivityModel": "Model",
    "superadmin.sensitivityExported": "Original export (reconstructed)",
    "superadmin.sensitivityRefitOld": "Refit with original preprocessing",
    "superadmin.sensitivityCorrected": "Refit with training-only medians",
    "superadmin.sensitivityDelta": "Change versus original refit",
    "superadmin.sensitivityMedians": "Medians changed by the correction",
    "superadmin.sensitivityNoMedianChanges": "No numeric fill medians changed.",
    "superadmin.createUser": "Create User",
    "superadmin.professionPlaceholder": "Enter a profession",
    "superadmin.inviteLink": "Invite Link",
    "superadmin.generateInvite": "Generate Invite",
    "superadmin.inviteCopied": "Invite link copied to clipboard.",
    "superadmin.expires": "Expires",
    "superadmin.inviteDescription":
      "Generate an invite link to allow new registrations.",
    "superadmin.chartDisplayTitle": "Chart Display",
    "superadmin.chartDisplayDescription":
      "Control global display options for shape-function charts and prediction results.",
    "superadmin.showMissingBars": "Show missing bars",
    "superadmin.hideMissingBars": "Hide missing bars",
    "superadmin.showCompetenceLevels": "Show competence table",
    "superadmin.hideCompetenceLevels": "Hide competence table",
    "superadmin.predictionComparisonVisibilityTitle":
      "Prediction Comparison visibility",
    "superadmin.predictionComparisonVisibilityDescription":
      "Choose who can see the Prediction Comparison section on the main page. Both views are hidden by default.",
    "superadmin.showPredictionComparisonSelf": "Show for me",
    "superadmin.hidePredictionComparisonSelf": "Hide for me",
    "superadmin.showPredictionComparisonUsers": "Show for other users",
    "superadmin.hidePredictionComparisonUsers": "Hide for other users",
    "superadmin.updatingChartDisplay": "Updating chart display...",
    "superadmin.error.updateChartDisplaySettings":
      "Chart display settings could not be updated",
    "superadmin.systemReset": "System Reset",
    "superadmin.systemResetDescription":
      "Permanently removes all users, all saved edits, uploaded CSV files, and extracted feature state for the current backend environment.",
    "superadmin.resetDatabase": "Reset Database",
    "superadmin.creating": "Creating...",
    "superadmin.users": "Users",
    "superadmin.loadingUsers": "Loading users...",
    "superadmin.noUsers": "No users yet.",
    "superadmin.created": "Created",
    "superadmin.badge": "Superadmin",
    "superadmin.resetConfirmTitle": "Reset Database?",
    "superadmin.resetConfirmBody":
      "This permanently deletes users, edits, uploaded CSV files, and extracted feature state. This action cannot be undone.",
    "superadmin.resetting": "Resetting...",
    "superadmin.resetEverything": "Yes, Reset Everything",
    "superadmin.modelTransferTitle": "Model Transfer",
    "superadmin.modelTransferDescription":
      "Export the active trained model to JSON or import a previously exported model artifact.",
    "superadmin.exporting": "Exporting...",
    "superadmin.exportModel": "Export Model",
    "superadmin.importing": "Importing...",
    "superadmin.importModel": "Import Model",
    "superadmin.modelTransferWarning":
      "Importing a model replaces the current base model and clears saved user edits tied to the previous one.",
    "superadmin.exportSuccess": "Model exported as {filename}.",
    "superadmin.exportSuccessWithEdits":
      "Model and saved shape-function edits exported as {filename}.",
    "superadmin.importSuccess": "Model imported successfully.",
    "superadmin.importSuccessWithEdits":
      "Model and saved shape-function edits imported successfully for {users} users across {submissions} submissions.",
    "superadmin.exportConfirmTitle": "Export Model Artifact?",
    "superadmin.exportConfirmBody":
      "Choose whether the exported model artifact should also include saved shape-function edits tied to the current model. The export also contains the held-out test rows and target values needed for later evaluation.",
    "superadmin.exportIncludeEdits":
      "Include saved shape-function edits",
    "superadmin.exportConfirmAction": "Export Artifact",
    "modelCompare.title": "Exported Model Comparison",
    "modelCompare.subtitle":
      "Upload two exported model JSON artifacts and compare their base and edited feature shapes.",
    "modelCompare.previewOnly": "Preview only - live model remains unchanged",
    "modelCompare.leftArtifact": "Full-data artifact",
    "modelCompare.rightArtifact": "Reduced-data artifact",
    "modelCompare.leftArtifactSelections": "Full-data artifact selections",
    "modelCompare.rightArtifactSelections": "Reduced-data artifact selections",
    "modelCompare.noFileChosen": "No file chosen yet",
    "modelCompare.compareArtifacts": "Compare artifacts",
    "modelCompare.instructions":
      "Both uploads must be exported model JSON files with the same target column and selected feature set.",
    "modelCompare.preparing": "Preparing comparison...",
    "modelCompare.prepareError": "Failed to prepare model comparison",
    "modelCompare.previewError": "Failed to recompute comparison preview",
    "modelCompare.chooseBothFiles":
      "Please choose both exported JSON files before comparing them.",
    "modelCompare.overlapTitle": "CSV overlap with artifact test sets",
    "modelCompare.overlapDescription":
      "Upload the reduced model's source CSV to check its actual training rows against the original test set. For exact verification when the artifacts lack shared source IDs, also upload the older full source CSV. A separate CSV row comparison is shown below.",
    "modelCompare.overlapCsvLabel": "Reduced-data CSV",
    "modelCompare.overlapSourceCsvLabel": "Older full source CSV (optional)",
    "modelCompare.overlapCheck": "Check overlap",
    "modelCompare.overlapChecking": "Checking overlap...",
    "modelCompare.overlapError": "Could not check CSV overlap",
    "modelCompare.overlapUnavailable":
      "The original artifact has no held-out test rows. Export a model with test data to run this check.",
    "modelCompare.overlapEditedUnavailable":
      "The edited artifact has no held-out test rows to subtract. The CSV will be compared with the original test set only.",
    "modelCompare.overlapResult":
      "{matches} of {remaining} remaining CSV rows match the original held-out test set ({testRows} test rows).",
    "modelCompare.overlapDirectResult":
      "{matches} of {remaining} CSV rows match the original held-out test set ({testRows} test rows).",
    "modelCompare.overlapSubtraction":
      "CSV rows: {csvRows}. Removed because they match the edited held-out set ({editedTestRows} test rows): {removed}. Remaining: {remaining}.",
    "modelCompare.overlapBeforeSubtraction":
      "Before subtraction, {matches} CSV rows matched the original held-out set.",
    "modelCompare.overlapWithTarget":
      "Matching used all model features and the target column.",
    "modelCompare.overlapFeaturesOnly":
      "The CSV has no original target column, so matching used model features only.",
    "modelCompare.overlapMatchedRows": "Remaining rows matching the original test set (numbered after the CSV header):",
    "modelCompare.overlapDirectMatchedRows": "Rows matching the original test set (numbered after the CSV header):",
    "modelCompare.overlapRemovedRows": "Removed CSV rows matching the edited test set ({count})",
    "modelCompare.overlapRowDetail":
      "CSV row {csvRow} matches held-out row {testRows}",
    "modelCompare.overlapSampleIds": "sample ID: {ids}",
    "modelCompare.overlapShowingFirst": "Showing the first {count} matching CSV rows.",
    "modelCompare.trainingOverlapTitle": "Reduced training rows in the original test set",
    "modelCompare.trainingOverlapExact":
      "{matches} of {training} reduced-model training observations occur in the original model's {test}-row held-out test set.",
    "modelCompare.trainingOverlapSourceVerified":
      "Verified by mapping complete reduced CSV rows to the older full source CSV and checking both artifact splits.",
    "modelCompare.trainingOverlapIdsVerified":
      "Verified using sample IDs from a shared source namespace in both artifacts.",
    "modelCompare.trainingOverlapTargetOnly":
      "{matches} of {training} reduced training rows share unique target values with the original {test}-row test set.",
    "modelCompare.trainingOverlapNeedsSource":
      "This target-value count does not prove the rows are the same observations. Upload the older full source CSV to verify the exact overlap.",
    "modelCompare.trainingOverlapUnavailable":
      "Training-to-test overlap could not be verified: {reason}",
    "modelCompare.trainingOverlapUsableRows":
      "Usable reduced-data CSV rows after excluding missing targets: {count}.",
    "modelCompare.overlapDescriptiveTitle":
      "Other CSV row matches (not the training leakage count)",
    "modelCompare.exportedAt": "Exported at",
    "modelCompare.dataset": "Dataset",
    "modelCompare.features": "Features",
    "modelCompare.editExport": "Edit export",
    "modelCompare.selectedSubmissions": "Selected submissions",
    "modelCompare.selectionHint":
      "Select or deselect imported user submissions to preview their effect.",
    "modelCompare.selected": "selected",
    "modelCompare.noImportedEdits":
      "This artifact does not include imported user edit submissions.",
    "modelCompare.noImportedEditsForFeature":
      "This artifact has no imported user edit submissions for this feature.",
    "modelCompare.confidence": "Confidence",
    "modelCompare.points": "points",
    "modelCompare.xSummary": "X summary",
    "modelCompare.chartControls": "Chart controls",
    "modelCompare.chartControlsDescription":
      "Switch between confidence-weighted and equal-weight preview, search features, and jump to a chart.",
    "modelCompare.confidenceOn": "Confidence weighting on",
    "modelCompare.confidenceOff": "Confidence weighting off",
    "modelCompare.showAlternateConfidenceAggregation":
      "Compare both confidence aggregations",
    "modelCompare.hideAlternateConfidenceAggregation":
      "Show selected aggregation only",
    "modelCompare.featureMetricsOriginalTestSet":
      "Use original test set for per-feature MAE, RMSE and R²",
    "modelCompare.featureMetricsUniformGridDescription":
      "When off, numeric metrics use 100 evenly spaced points across the shared curve range.",
    "modelCompare.featureMetricsOriginalTestSetUnavailable":
      "Requires held-out feature rows in the original artifact.",
    "modelCompare.confidenceDescription":
      "When off, selected submissions are averaged equally instead of using their saved confidence.",
    "modelCompare.showBaseTraces": "Show base lines and bars",
    "modelCompare.hideBaseTraces": "Hide base lines and bars",
    "modelCompare.centerCharts": "Center curves",
    "modelCompare.restoreChartPositions": "Restore curve positions",
    "modelCompare.showLineDots": "Show dots on lines",
    "modelCompare.hideLineDots": "Hide dots on lines",
    "modelCompare.makeChartsTaller": "Make charts taller",
    "modelCompare.restoreChartHeight": "Restore chart height",
    "modelCompare.baseTracesDescription":
      "Toggle the original baseline traces for both compared artifacts.",
    "modelCompare.showIndividualEdits": "Show individual edits",
    "modelCompare.showAggregatedEdits": "Show aggregated edits",
    "modelCompare.enableChartExpertSelection": "Select experts in feature charts",
    "modelCompare.disableChartExpertSelection": "Stop selecting experts in feature charts",
    "modelCompare.chartExpertSelectionHint":
      "Choose experts to display in this feature's shape and MAE-change charts. Other results stay unchanged.",
    "modelCompare.showIndividualMae": "Show individual MAE errors",
    "modelCompare.hideIndividualMae": "Hide individual MAE errors",
    "modelCompare.individualMaeErrors": "Individual participant MAE errors",
    "modelCompare.showIndividualMaeChangeChart": "Show expert MAE-change chart",
    "modelCompare.hideIndividualMaeChangeChart": "Hide expert MAE-change chart",
    "modelCompare.showConfidenceRatedMae": "Show confidence-rated expert MAE",
    "modelCompare.hideConfidenceRatedMae": "Hide confidence-rated expert MAE",
    "modelCompare.expertMaeChange": "Expert MAE change",
    "modelCompare.expertMaeChangeTitle": "Change in shape-function MAE by expert",
    "modelCompare.expertMaeChangeAxis": "MAE Δ from reduced-data baseline",
    "modelCompare.allFeatureMaeChange":
      "Aggregated MAE change across all features",
    "modelCompare.allFeatureMaeChangeDescription":
      "Each bar shows one feature after all exported edits, compared with the unedited models. Both confidence modes ignore expert selections in feature charts. In the combined view, patterned bars include confidence ratings.",
    "modelCompare.expertAverageConfidence": "Average confidence by expert",
    "modelCompare.expertAverageConfidenceDescription":
      "Average saved confidence rating across all exported feature submissions for each expert. Bars are ordered from highest to lowest and do not change with feature chart selections.",
    "modelCompare.expertAverageConfidenceAxis": "Average confidence rating (1–10)",
    "modelCompare.expertAverageConfidenceRatings": "Rated submissions",
    "modelCompare.noExpertConfidenceRatings": "No saved expert confidence ratings are available.",
    "modelCompare.allFeatureMaeChangeWithoutConfidence":
      "Without confidence rating",
    "modelCompare.allFeatureMaeChangeWithConfidence":
      "With confidence rating included",
    "modelCompare.showCombinedMaeChart": "Show combined chart",
    "modelCompare.showSeparateMaeCharts": "Show separate charts",
    "modelCompare.showConfidenceInLegend": "Show confidence rating in legend",
    "modelCompare.hideConfidenceInLegend": "Hide confidence rating in legend",
    "modelCompare.individualEditsDescription":
      "Show one solid trace for every selected edit instead of the combined trace; colors identify experts.",
    "modelCompare.legendRightOfChart": "Move legend to the right",
    "modelCompare.legendAboveChart": "Move legend above chart",
    "modelCompare.legendPositionDescription":
      "Place every chart legend to the right of its plot or in the default position above it.",
    "modelCompare.searchFeatures": "Search features",
    "modelCompare.jumpToFeature": "Jump to feature",
    "modelCompare.featureCharts": "Feature charts",
    "modelCompare.featureChartsDescription":
      "Dashed traces show each artifact's base shape. Solid traces show either the combined preview or each selected edit.",
    "modelCompare.overallShapeFunctionMae": "Overall shape-function MAE",
    "modelCompare.overallShapeFunctionMaeDescription":
      "Compares the reduced-data baseline and selected edits with the full-data shape functions. Lower is closer.",
    "modelCompare.overallShapeFunctionMaeCoverage":
      "{features} shape functions · {points} reference points",
    "modelCompare.pooledShapeFunctionMae": "Pooled curve points",
    "modelCompare.meanPerFeatureMae": "Equal-feature mean",
    "modelCompare.expertCompleteModelMae": "Expert complete-model MAE",
    "modelCompare.expertCompleteModelMaeDescription":
      "Each value compares one expert's selected edits across all compared shape functions with the full-data curves.",
    "modelCompare.expertCompleteModelMaeConfidenceWeighted":
      "Saved confidence ratings are applied to the expert edits.",
    "modelCompare.expertCompleteModelMaeEqualWeighted":
      "Expert edits are averaged equally; saved confidence ratings are not applied.",
    "modelCompare.expertCompleteModelMaeCoverage":
      "{features} shape functions · {points} reference points · {submissions} submissions",
    "modelCompare.overallMaeFeatureSelection":
      "Choose shape functions ({selected} of {total} selected)",
    "modelCompare.overallMaeFeatureSelectionDescription":
      "Only checked shape functions are included in the overall MAE cards above. At least one must remain selected.",
    "modelCompare.featureMaeDelta": "MAE Δ: {delta}",
    "modelCompare.heldOutTestPerformance": "Held-out test performance",
    "modelCompare.heldOutTestPerformanceDescription":
      "Evaluates the full-data benchmark, original reduced-data model, and edited reduced-data model against the same held-out targets.",
    "modelCompare.model": "Model",
    "modelCompare.fullData": "Full-data benchmark",
    "modelCompare.reducedData": "Original reduced-data model",
    "modelCompare.editedReducedData": "Edited reduced-data model",
    "modelCompare.effectOfExpertEdits": "Effect of expert edits",
    "modelCompare.selectedModelComparison": "Selected model comparison",
    "modelCompare.reducedToEdited": "Original reduced-data → edited reduced-data",
    "modelCompare.fullToReduced": "Full-data benchmark → original reduced-data model",
    "modelCompare.fullToEdited": "Full-data benchmark → edited reduced-data model",
    "modelCompare.allThreeComparison": "Full-data benchmark → original reduced-data model → edited reduced-data model",
    "modelCompare.reducedMinusFull": "reduced − full",
    "modelCompare.editedMinusFull": "edited − full",
    "modelCompare.editedMinusReduced": "edited − reduced",
    "modelCompare.allThree": "All three",
    "modelCompare.reducedEdited": "Reduced vs. edited",
    "modelCompare.fullReduced": "Full vs. reduced",
    "modelCompare.fullEdited": "Full vs. edited",
    "modelCompare.residualComparison": "Actual value vs. residual",
    "modelCompare.residualAxis": "Prediction − actual",
    "modelCompare.zeroResidual": "Zero residual",
    "modelCompare.metric.mae": "MAE",
    "modelCompare.metric.rmse": "RMSE",
    "modelCompare.metric.r2": "R²",
    "modelCompare.metric.mean_error": "Mean error",
    "modelCompare.metric.bias_magnitude": "Absolute mean prediction error",
    "modelCompare.metric.std_dev": "Prediction standard deviation",
    "modelCompare.meanErrorConvention": "Mean error = mean(prediction − actual). Negative means underprediction; positive means overprediction. Comparison cards use the absolute mean prediction error.",
    "modelCompare.improved": "Improved",
    "modelCompare.worsened": "Worsened",
    "modelCompare.unchanged": "Unchanged",
    "modelCompare.benchmarkRelationships": "Full ↔ reduced: limited-data gap. Full ↔ edited: remaining gap after editing. Accuracy is measured against actual targets, not agreement with the benchmark.",
    "modelCompare.benchmarkGaps": "Difference from full-data benchmark",
    "modelCompare.observationEffects": "Per-observation edit effects and uncertainty",
    "modelCompare.medianErrorChange": "Median absolute-error change",
    "modelCompare.trainingCorrectionOption": "Apply training-only intercept correction to match the reduced model's mean prediction",
    "modelCompare.manualAdjustmentDisabled": "Training-only correction is active. Clear its checkbox in the performance section to enter a manual shift.",
    "modelCompare.correctionMetadata": "Edited intercept correction enabled: {enabled}; shift: {shift}; source: {source}.",
    "modelCompare.beforeAfterCorrection": "Edited model before and after intercept correction",
    "modelCompare.beforeCorrection": "Before correction",
    "modelCompare.afterCorrection": "After correction",
    "modelCompare.splitValidation": "Split check: the exported full and reduced test rows match: {shared}. Matching source sample IDs are unavailable for both training sets, so test exclusion cannot be proven from these artifacts. Matching feature rows in exported training data: full {full}, reduced {reduced}. Matches can also be distinct observations.",
    "modelCompare.splitVerified": "Split verified for {samples} test rows: supplied source sample IDs are disjoint from both models' training IDs.",
    "modelCompare.preprocessingLeakage": "Preprocessing caveat: these historical models calculated missing-value fills before the train/test split. Test feature values influenced preprocessing; test targets were not used to fit the models or intercept correction.",
    "modelCompare.mainTrainingPerformance": "Training-set performance (debug)",
    "modelCompare.mainTrainingPerformanceDescription":
      "Uses only the dataset rows that trained the main-page model. Validation and test rows are excluded; these in-sample metrics are for debugging.",
    "modelCompare.heldOutTestDataSource": "Test set",
    "modelCompare.useOriginalArtifactTestSet": "Full-data held-out test set",
    "modelCompare.useReducedArtifactTestSet": "Reduced artifact test set",
    "modelCompare.useMainTrainingData": "Full-data set",
    "modelCompare.evaluateHeldOutTest": "Evaluate artifacts",
    "modelCompare.heldOutTestCoverage": "{samples} held-out test samples",
    "modelCompare.sharedHeldOutNeedsLineage": "For this original IQB export, upload the reduced 5% CSV and old 100% source CSV in the overlap section, then run the overlap check before evaluating. The original 5,249-row test set includes reduced-model training rows.",
    "modelCompare.sharedHeldOutReady": "Verified shared held-out set: {samples} source rows, excluding reduced-model training observations.",
    "modelCompare.mainTrainingPerformanceCoverage": "{samples} training samples",
    "modelCompare.heldOutBeforeEdits": "Original artifact",
    "modelCompare.heldOutAfterEdits": "Edited artifact",
    "modelCompare.heldOutDeltaEditedMinusOriginal": "Edited − original",
    "modelCompare.heldOutResidualStandardDeviation": "Residual standard deviation",
    "modelCompare.heldOutMeanError": "Mean error",
    "modelCompare.actualVsPredicted": "Actual vs. predicted",
    "modelCompare.actualValues": "Actual values",
    "modelCompare.predictedValues": "Predicted values",
    "modelCompare.actualVsPredictedIdentityLine": "Perfect prediction",
    "modelCompare.heldOutTestReady":
      "Select a test set, then evaluate all three model states.",
    "modelCompare.heldOutTestUnavailable":
      "The selected artifact has no held-out test data. Export it again after training with this version, then upload the new artifact.",
    "modelCompare.averagePredictedScore":
      "Average prediction comparison",
    "modelCompare.averagePredictedScoreDescription":
      "Compares the full-data model, the unedited reduced-data model, and the edited reduced-data model on the selected reference; selected edits and confidence weighting are included.",
    "modelCompare.interceptAdjustmentLabel": "Edited model intercept adjustment",
    "modelCompare.interceptAdjustmentDescription":
      "Adds this constant to edited predictions, including the held-out performance test. Apply the 80% training-row suggestion above if you want to cancel the measured edit shift.",
    "modelCompare.applyInterceptAdjustment": "Apply",
    "modelCompare.interceptAdjustmentInvalid": "Enter a finite number for the intercept adjustment.",
    "modelCompare.originalEditedBaseOffset": "Original edited base offset",
    "modelCompare.activeInterceptAdjustment": "Active adjustment",
    "modelCompare.adjustedEditedBaseOffset": "Adjusted edited base offset",
    "modelCompare.testSetAverage": "Original held-out test set",
    "modelCompare.mainDatasetAverage": "Original model's 80% training set",
    "modelCompare.testSetAverageDescription":
      "Both artifacts predict every held-out feature row from the original model JSON; the cards show the mean predictions.",
    "modelCompare.mainDatasetAverageDescription":
      "Both artifacts predict the original model's {count} exported training rows (its 80% development split). The mean edit shift is calculated before the intercept adjustment.",
    "modelCompare.mainDatasetAverageUnavailable":
      "The original model artifact has no exported training rows. Export it again to calculate the 80% training-set shift.",
    "modelCompare.developmentMeanEditShift":
      "Mean edit shift before intercept adjustment",
    "modelCompare.developmentSuggestedAdjustment":
      "Suggested intercept adjustment from these training rows",
    "modelCompare.heldOutConfidenceWeighted":
      "Held-out predictions use confidence-weighted expert edits.",
    "modelCompare.heldOutUnweighted":
      "Held-out predictions use unweighted expert edits at their submitted strength.",
    "modelCompare.missingIndicatorsExcludedFromAverage":
      "Generated [missing] indicator features are excluded from these averages and their feature breakdown.",
    "modelCompare.testSetAverageUnavailable":
      "The original model JSON has no held-out test rows. Export it again after training to use this option.",
    "modelCompare.limitEditedMeanDeviation":
      "Limit edited mean shift to 10%",
    "modelCompare.limitEditedMeanDeviationDescription":
      "Scales every selected edit on the edited artifact by one shared factor; saved edits are unchanged.",
    "modelCompare.editScaleFactor": "×{factor}",
    "modelCompare.meanDeviationScaled":
      "Selected edited offsets are shown at {percent}% of their submitted magnitude to meet the limit.",
    "modelCompare.meanDeviationWithinLimit":
      "The selected edits are already within the 10% limit.",
    "modelCompare.meanDeviationCannotMeetLimit":
      "The unedited right-hand model already differs by more than 10%, so scaling edits alone cannot meet the limit.",
    "modelCompare.originalAveragePredictedScore": "Full-data original artifact",
    "modelCompare.editedAveragePredictedScore":
      "Edited artifact on selected reference",
    "modelCompare.averagePredictedScoreDifference": "Edited − full-data original",
    "modelCompare.editedMinusReducedAveragePrediction":
      "Average prediction difference: edited reduced model − unedited reduced model",
    "modelCompare.averagePredictionRelativeChange": "Relative change: {change}",
    "modelCompare.trainingMeanDetailsTitle": "Training mean and intercept values",
    "modelCompare.trainingMeanDetailsDescription":
      "Mean predictions reconstructed from the artifacts on the reduced-data model's preprocessed training rows ({count} rows), excluding generated [missing] indicators. Selected expert adaptations and preview scaling are included. The after-correction value uses the currently applied intercept adjustment.",
    "modelCompare.trainingMeanUnavailable":
      "Training means are unavailable because this artifact does not contain training feature rows. Re-export the reduced-data model with this version and upload it again.",
    "modelCompare.reducedTrainingMean": "Reduced-data model mean prediction",
    "modelCompare.editedTrainingMeanBeforeCorrection":
      "Expert-adapted mean before intercept correction",
    "modelCompare.trainingMeanDifferenceBeforeCorrection":
      "Expert adaptation difference (edited − reduced)",
    "modelCompare.editedTrainingMeanAfterCorrection":
      "Expert-adapted mean after applied correction",
    "modelCompare.originalFullDataIntercept": "Original full-data intercept",
    "modelCompare.reducedModelIntercept": "Reduced-data model intercept",
    "modelCompare.editedInterceptBeforeCorrection":
      "Expert-adapted intercept before correction",
    "modelCompare.appliedInterceptCorrection": "Applied intercept correction",
    "modelCompare.editedInterceptAfterCorrection":
      "Expert-adapted intercept after applied correction",
    "modelCompare.suggestedMatchingCorrection":
      "Correction to match reduced-data training mean",
    "modelCompare.suggestedMatchingIntercept":
      "Intercept with matching correction",
    "modelCompare.averageContributionsTitle":
      "Change in average prediction by feature",
    "modelCompare.reducedEditedContributionsTitle":
      "Change in average prediction by feature: edited reduced − original reduced model",
    "modelCompare.contributionFeature": "Feature",
    "modelCompare.meanPredictionChangeColumn":
      "Mean prediction change (edited − original)",
    "modelCompare.reducedEditedMeanPredictionChangeColumn":
      "Mean prediction change (edited reduced − original reduced)",
    "modelCompare.totalMeanDeviationFromOriginal":
      "Total mean deviation from original full-data model",
    "modelCompare.totalMeanDeviationFromOriginalReduced":
      "Total mean deviation from original reduced model",
    "modelCompare.featureMeanShiftLimit": "10% mean limit",
    "modelCompare.meanShiftFeatureLimitDescription":
      "Choose the edited features to scale. One optimal factor is calculated for that set; all other feature edits stay unchanged.",
    "modelCompare.chooseMeanShiftFeatures": "Choose features",
    "modelCompare.meanShiftFeatureModalTitle":
      "Choose features for the 10% mean limit",
    "modelCompare.meanShiftFeatureModalDescription":
      "The selected features receive one optimal shared factor so the selected overall average moves into the 10% range.",
    "modelCompare.meanShiftFeatureModalSelection":
      "{selected} of {total} features selected",
    "modelCompare.applyMeanShiftFeatureLimit": "Apply optimal factor",
    "modelCompare.limitFeatureMeanShift": "Limit this feature",
    "modelCompare.removeFeatureMeanShiftLimit": "Remove feature limit",
    "modelCompare.selectedContributionAverageView":
      "Cards show the base offset plus selected feature contributions.",
    "modelCompare.recomputing": "Recomputing preview...",
    "modelCompare.noMatchingFeatures":
      "No feature charts match the current search.",
    "modelCompare.maeVsEditedBase": "Edited vs edited base",
    "modelCompare.maeVsOriginalBase": "Edited vs original base",
    "modelCompare.rmseVsEditedBase": "Edited vs edited base",
    "modelCompare.rmseVsOriginalBase": "Edited vs original base",
    "modelCompare.r2VsEditedBase": "Edited vs edited base",
    "modelCompare.r2VsOriginalBase": "Edited vs original base",
    "modelCompare.maeOriginalVsEdited": "Original vs selected edits",
    "modelCompare.rmseOriginalVsEdited": "Original vs selected edits",
    "modelCompare.r2OriginalVsEdited": "Original vs selected edits",
    "modelCompare.metricImproved": "Improved",
    "modelCompare.metricWorsened": "Worsened",
    "modelCompare.metricUnchanged": "Unchanged",
    "modelCompare.reducedVsFullData": "Reduced-data vs full-data",
    "modelCompare.deltaFromBaseline": "Δ from reduced-data baseline",
    "modelCompare.relativeChange": "Relative change",
    "modelCompare.leftBase": "Full-data baseline",
    "modelCompare.leftEffective": "Original",
    "modelCompare.rightBase": "Reduced-data baseline",
    "modelCompare.rightEffective": "Edited",
    "modelCompare.aggregatedParticipantEdits": "Aggregated expert edits",
    "modelCompare.editLeftSelections": "Edit original selections",
    "modelCompare.editRightSelections": "Edit edited selections",
  },
  de: {
    "language.switchToEnglish": "Sprache auf Englisch umstellen",
    "language.switchToGerman": "Sprache auf Deutsch umstellen",

    "common.yes": "Ja",
    "common.no": "Nein",
    "common.cancel": "Abbrechen",
    "common.username": "Benutzername",
    "common.password": "Passwort",
    "common.profession": "Beruf",
    "common.loading": "Laden...",
    "common.refresh": "Aktualisieren",
    "common.save": "Speichern",
    "common.apply": "Anwenden",
    "common.reset": "Zurücksetzen",
    "common.close": "Schließen",
    "common.delete": "Löschen",
    "common.target": "Ziel",
    "common.unknown": "Unbekannt",

    "header.subtitle": "Interaktiver GAM-Editor",
    "header.modelTrained": "Modell trainiert",
    "header.modelNotTrained": "Modell nicht trainiert",
    "header.recordsLoaded": "{count} Datensätze geladen",

    "login.title": "Der interaktive GAM-Editor",
    "login.subtitleLogin": "Mit deinen Zugangsdaten anmelden",
    "login.subtitleRegister": "Mit einem Einladungslink registrieren",
    "login.usernamePlaceholder": "Dein Benutzername",
    "login.passwordPlaceholder": "Dein Passwort",
    "login.professionPlaceholder": "Dein Beruf",
    "login.registerInviteOnly":
      "Eine Registrierung ist nur über einen Einladungslink möglich.",
    "login.error.enterCredentials":
      "Bitte Benutzername und Passwort eingeben",
    "login.error.professionRequired": "Bitte gib deinen Beruf ein",
    "login.error.inviteRequired":
      "Die Registrierung erfordert einen gültigen Einladungslink",
    "login.error.loginFailed": "Anmeldung fehlgeschlagen",
    "login.error.registerFailed": "Registrierung fehlgeschlagen",
    "login.error.serverUnavailable":
      "Der Server ist nicht erreichbar. Bitte versuche es erneut.",
    "login.submitSigningIn": "Anmeldung läuft...",
    "login.submitRegistering": "Registrierung läuft...",
    "login.submitSignIn": "Anmelden",
    "login.submitRegister": "Registrieren",

    "app.loggedInAs": "Angemeldet als",
    "app.viewCombinedResults": "Kombinierte Ergebnisse anzeigen",
    "app.superadmin": "Superadmin",
    "app.logout": "Abmelden",
    "app.footerDescription":
      "GAMWeaver - Interaktiver GAM-Editor für interaktive Modellierung und Erklärbarkeit. Entwickelt von {name} im Rahmen einer Masterarbeit.",
    "app.githubLink": "GAMWeaver GitHub",
    "app.notification.editRemoved": "Bearbeitung entfernt",
    "app.notification.editsRemoved": "Bearbeitungen entfernt",
    "app.notification.singleMessage":
      "Eine deiner Bearbeitungen wurde von einer anderen Person entfernt:",
    "app.notification.multipleMessage":
      "{count} deiner Bearbeitungen wurden von anderen Personen entfernt:",
    "app.notification.feature": "Merkmal",
    "app.notification.by": "von",
    "app.notification.curveEdit": "Kurvenbearbeitung",
    "app.notification.pointCount": "Punkte",
    "app.notification.xSummary": "X-Bereich",
    "app.notification.xValue": "X-Wert",
    "app.notification.reason": "Grund",
    "app.notification.gotIt": "Verstanden",
    "app.error.applyShapeEdits":
      "Bearbeitungen der Shape-Funktionen konnten nicht angewendet werden",
    "app.error.resetShapeFunctions":
      "Shape-Funktionen konnten nicht zurückgesetzt werden",
    "app.error.resetFeature":
      "Feature konnte nicht zurückgesetzt werden",
    "app.error.updateFeatureChartSettings":
      "Feature-Diagrammeinstellungen konnten nicht aktualisiert werden",
    "app.error.loadData": "Daten konnten nicht geladen werden",
    "app.error.uploadDataset":
      "Datensatz konnte nicht hochgeladen werden",
    "app.error.trainModel":
      "Modell konnte nicht trainiert werden",
    "app.error.predictionFailed": "Vorhersage fehlgeschlagen",

    "training.title": "Modelltraining",
    "training.restrictedNotice":
      "Das Hochladen/Laden von Datensätzen und die Trainingssteuerung sind auf den Superadmin beschränkt.",
    "training.dataLoaded": "Daten geladen",
    "training.modelTrainedLabel": "Modell trainiert",
    "training.activeDataset": "Aktiver Datensatz",
    "training.stepLoadData": "Daten laden",
    "training.stepLoaded": "Geladen",
    "training.loadDescription":
      "CSV-Datensatz hochladen, Vorhersageziel auswählen und zu importierende Spalten festlegen.",
    "training.uploadAndLoad": "Datensatz hochladen und laden",
    "training.stepTrainModel": "Modell trainieren",
    "training.stepTrained": "Trainiert",
    "training.trainDescription": "IGANN-Modell konfigurieren und trainieren.",
    "training.estimators": "Anzahl der Estimatoren",
    "training.estimatorHint":
      "Mehr Estimatoren = bessere Genauigkeit, aber langsameres Training",
    "training.trainButton": "IGANN-Modell trainieren",
    "training.uploadModalTitle": "Datensatz hochladen",
    "training.csvFile": "CSV-Datei",
    "training.useCaseStudyFeatureNames":
      "Fallspezifische Feature-Namen verwenden",
    "training.uploadInspect": "Hochladen und Spalten prüfen",
    "training.columnsDetected": "Erkannte Spalten",
    "training.targetColumn": "Zielspalte",
    "training.featureColumns": "Feature-Spalten",
    "training.expandFeatureColumns": "Feature-Spalten erweitern",
    "training.collapseFeatureColumns": "Feature-Spalten verkleinern",
    "training.selectAll": "Alle wählen",
    "training.clearAll": "Auswahl löschen",
    "training.target": "Ziel",
    "training.selectedFeatures": "Ausgewählte Features",
    "training.loadDataset": "Datensatz laden",
    "training.error.chooseCsvFirst": "Bitte zuerst eine CSV-Datei auswählen.",
    "training.error.uploadFailed": "Datensatz konnte nicht hochgeladen werden",
    "training.error.loadFailed": "Datensatz konnte nicht geladen werden",
    "training.error.uploadNeedsSelections":
      "Bitte eine CSV hochladen, eine Zielspalte wählen und mindestens eine Feature-Spalte auswählen.",
    "training.progress.loadingData": "Daten werden geladen...",
    "training.progress.training":
      "Modell wird trainiert... Das kann einen Moment dauern.",
    "training.uploading": "Wird hochgeladen...",
    "training.loadingDataset": "Wird geladen...",
    "training.training": "Training läuft...",
    "training.statusDataLoaded": "Daten geladen",
    "training.statusModelTrained": "Modell trainiert",
    "training.statusModelSource": "Modellquelle",
    "training.modelSourceImported": "importiert",
    "training.modelSourceTrained": "trainiert",
    "training.importedAnalyticsUnavailable":
      "Analysen für importierte Modelle bleiben nicht verfügbar, bis der Superadmin einen kompatiblen Datensatz lädt.",

    "dataSummary.title": "Datenübersicht",
    "dataSummary.empty": "Daten laden, um Zusammenfassungsstatistiken zu sehen",
    "dataSummary.totalRecords": "Gesamtanzahl Datensätze",
    "dataSummary.targetVariable": "Zielvariable",
    "dataSummary.targetFallback": "Ziel",
    "dataSummary.mean": "Mittelwert",
    "dataSummary.stdDev": "Std.-Abw.",
    "dataSummary.min": "Min",
    "dataSummary.max": "Max",
    "dataSummary.features": "Features",

    "prediction.title": "Vorhersage erstellen",
    "prediction.description":
      "Feature-Werte des Datensatzes setzen und eine Vorhersage ausführen. Die Eingaben werden aus dem aktiven Datensatzschema generiert.",
    "prediction.trainFirst":
      "Bitte zuerst das Modell trainieren, bevor Vorhersagen erstellt werden.",
    "prediction.noSchema":
      "Kein Feature-Schema verfügbar. Bitte den Superadmin, einen Datensatz zu laden.",
    "prediction.predicting": "Vorhersage läuft...",
    "prediction.predictTarget": "{target} vorhersagen",
    "prediction.predictedTarget": "Vorhergesagtes {target}",
    "prediction.scaleHelpLabel":
      "Werteskala für {feature} erklären",
    "prediction.scaleTooltipOptionsIntro":
      "Diese Werte verwenden dieselbe X-Achsen-Kodierung:",
    "prediction.scaleTooltipAvailableValues":
      "Verfügbare X-Achsen-Werte: {values}.",
    "prediction.scaleTooltipRange":
      "Diese Eingabe verwendet dieselbe X-Achsen-Skala: {min} bis {max}.",
    "prediction.scaleTooltipSingleValue":
      "Diese Eingabe verwendet den festen X-Achsen-Wert {value}.",
    "prediction.competenceLevel": "Kompetenzstufe",
    "prediction.scoreValue": "Punktwert",
    "prediction.assignment": "Zuordnung",
    "prediction.target.germanReading": "Lesekompetenz Deutsch",
    "prediction.option.TR_TLS1.0": "Nein",
    "prediction.option.TR_TLS1.1": "Ja",
    "prediction.option.KBuecher.1": "0–10 Bücher",
    "prediction.option.KBuecher.2": "11–25 Bücher",
    "prediction.option.KBuecher.3": "26–100 Bücher",
    "prediction.option.KBuecher.4": "101–200 Bücher",
    "prediction.option.KBuecher.5": "201–500 Bücher",
    "prediction.option.KBuecher.6": "Mehr als 500 Bücher",
    "prediction.option.KDezh16_r.1": "Spricht zu Hause nie Deutsch",
    "prediction.option.KDezh16_r.2":
      "Spricht zu Hause Deutsch und eine andere Sprache",
    "prediction.option.KDezh16_r.3":
      "Spricht zu Hause immer oder fast immer Deutsch",

    "shapeFunctions.title": "Shape-Funktionen",
    "shapeFunctions.emptyDescription":
      "Trainiere das Modell, um Feature-Shape-Funktionen zu sehen.",
    "shapeFunctions.sharedDescription":
      "Shape-Funktionen zeigen, wie jedes Feature die Vorhersage beeinflusst.",
    "shapeFunctions.gridDescription":
      "Diese Diagramme zeigen, wie jedes Feature die Vorhersage der Fahrradvermietung beeinflusst. Werte über null erhöhen die Vorhersage, Werte darunter senken sie.",
    "shapeFunctions.effectOnPrediction":
      "Effekt auf die Vorhersage",
    "shapeFunctions.interactiveTitle": "Interaktive Shape-Funktionen",
    "shapeFunctions.interactiveEmptyDescription":
      "Trainiere das Modell, um Feature-Shape-Funktionen zu sehen und zu bearbeiten.",
    "shapeFunctions.interactiveEditDescription":
      "Bearbeite Punkte und klicke dann bei jedem Feature auf Senden, um mit deiner Sicherheit zu speichern.",
    "shapeFunctions.interactiveViewDescription":
      "Aktiviere den Bearbeitungsmodus, um Shape-Funktionen interaktiv zu ändern.",
    "shapeFunctions.resetAll": "Alles zurücksetzen",
    "shapeFunctions.syncAxes": "Achsen synchronisieren",
    "shapeFunctions.axesSynced": "Achsen synchronisiert",
    "shapeFunctions.centerCurves": "Kurven zentrieren",
    "shapeFunctions.centerCurvesHint":
      "Nur für diese Ansicht den Mittelwert jeder Kurve abziehen.",
    "shapeFunctions.syncAxesHintOn":
      "Achsen sind synchronisiert - klicken für individuelle Skalierung",
    "shapeFunctions.syncAxesHintOff":
      "Klicken, um alle Diagrammachsen auf dieselbe Skala zu synchronisieren",
    "shapeFunctions.enableEditing": "Bearbeitung aktivieren",
    "shapeFunctions.editingModeOn": "Bearbeitungsmodus AN",
    "shapeFunctions.editHelp":
      "Bei numerischen Features klicken und ziehen, um Bearbeitungen entlang der Kurve zu pinseln. Mit dem Weich-Hart-Regler steuerst du, wie viel der Kurve beeinflusst wird. Bei kategorialen Features einen Punkt anfahren und nach oben oder unten ziehen. Doppelklick für eine präzise Eingabe. Wenn du mit einem Feature fertig bist, klicke auf Senden und bewerte deine Sicherheit.",
    "shapeFunctions.lineBrushHardness": "Pinselhärte der Linie",
    "shapeFunctions.soft": "Weich",
    "shapeFunctions.hard": "Hart",
    "shapeFunctions.original": "Original",
    "shapeFunctions.edited": "Bearbeitet",
    "shapeFunctions.current": "Aktuell",
    "shapeFunctions.comparison": "Vergleich",
    "shapeFunctions.missing": "Fehlend",
    "shapeFunctions.editPoints": "Punkte bearbeiten",
    "shapeFunctions.effect": "Effekt",
    "shapeFunctions.draggingSuffix": "(wird gezogen...)",
    "shapeFunctions.editedSuffix": "(bearbeitet)",
    "shapeFunctions.setEffectValue": "Effektwert festlegen",
    "shapeFunctions.brushingAtX": "Pinsel bei x = {value}",
    "shapeFunctions.releaseToApply":
      "Loslassen, um geglätteten Strich anzuwenden: {value}",
    "shapeFunctions.draggingPoint": "Punkt {index} wird gezogen",
    "shapeFunctions.releaseToSetValue":
      "Loslassen, um Wert zu setzen: {value}",
    "shapeFunctions.clickDragHint":
      "Klicken und ziehen, um die Linie zu bearbeiten. Doppelklick für eine präzise Eingabe.",
    "shapeFunctions.pointSelected": "Punkt {index} ausgewählt",
    "shapeFunctions.preciseValueHint":
      "Doppelklick für präzisen Wert",
    "shapeFunctions.hoverSelectHint":
      "Mit der Maus über einen Punkt fahren, um ihn auszuwählen",
    "shapeFunctions.submitEditTitle":
      "Bearbeitung für {feature} senden",
    "shapeFunctions.submitEditDescription":
      "Bewerte deine Sicherheit und beschreibe deine Bearbeitung.",
    "shapeFunctions.confidenceLevel": "Sicherheitsstufe",
    "shapeFunctions.notSure": "Unsicher",
    "shapeFunctions.verySure": "Sehr sicher",
    "shapeFunctions.editDescription":
      "Beschreibung der Bearbeitung",
    "shapeFunctions.editDescriptionPlaceholder":
      "Beschreibe, warum du diese Bearbeitung vorgenommen hast...",
    "shapeFunctions.editDescriptionRequired":
      "Bitte gib eine Beschreibung für deine Bearbeitung ein.",
    "shapeFunctions.submitEdit": "Bearbeitung senden",
    "shapeFunctions.chartSettingsSaveError":
      "Diagrammeinstellungen konnten nicht gespeichert werden",
    "shapeFunctions.chartMappingTitle":
      "Diagrammzuordnung: {feature}",
    "shapeFunctions.chartMappingDescription":
      "Konfiguriere Diagrammtyp und X-Achsen-Beschriftungen für alle Benutzer.",
    "shapeFunctions.displayTitleLabel": "Anzeigetitel",
    "shapeFunctions.displayTitleHint":
      "Nur kosmetisch. Diagramme und UI-Beschriftungen nutzen diesen Titel, Exporte behalten jedoch den ursprünglichen Featurenamen.",
    "shapeFunctions.xAxisLabel": "X-Achsenbeschriftung",
    "shapeFunctions.xAxisLabelHint":
      "Optional. Leer lassen, um den Anzeigetitel auf der X-Achse zu verwenden.",
    "shapeFunctions.chartType": "Diagrammtyp",
    "shapeFunctions.numericChartType": "Numerisch",
    "shapeFunctions.categoricalChartType": "Kategorial",
    "shapeFunctions.numericMappingHint":
      "Weise ganzzahligen numerischen X-Achsen-Werten Anzeigelabels zu, ohne die kontinuierliche Achse zu verändern.",
    "shapeFunctions.categoricalMappingHint":
      "Weise den auf diesem Diagramm gezeigten kategorialen X-Achsen-Werten Anzeigelabels zu.",
    "shapeFunctions.treatAsCategorical":
      "Dieses Diagramm als kategorial behandeln",
    "shapeFunctions.treatAsCategoricalHint":
      "Balken mit diskreten Kategorien statt einer kontinuierlichen Linie verwenden.",
    "shapeFunctions.cannotConvertCategorical":
      "Dieses Feature hat zu viele oder nicht-diskrete Werte und kann nicht in ein kategoriales Diagramm umgewandelt werden.",
    "shapeFunctions.xAxisValueLabels":
      "Wertebeschriftungen der X-Achse",
    "shapeFunctions.numericXAxisLabels":
      "Ganzzahlige X-Achsen-Beschriftungen",
    "shapeFunctions.noCategoricalValues":
      "Keine kategorialen Werte für die Zuordnung verfügbar.",
    "shapeFunctions.noNumericValues":
      "Keine numerischen X-Achsen-Werte für die Zuordnung verfügbar.",
    "shapeFunctions.displayLabelFor":
      "Anzeigelabel für {value}",
    "shapeFunctions.mappingAvailableAfterCategorical":
      "Label-Zuordnungen sind verfügbar, sobald dieses Feature als kategoriales Diagramm angezeigt wird.",
    "shapeFunctions.saveMapping": "Zuordnung speichern",
    "shapeFunctions.saving": "Speichern...",
    "shapeFunctions.switchChartToNumeric":
      "Diagramm auf numerisch umstellen",
    "shapeFunctions.switchChartToCategorical":
      "Diagramm auf kategorial umstellen",
    "shapeFunctions.chartTypeUnavailable":
      "Dieses Feature kann nicht in den anderen Diagrammtyp umgeschaltet werden",
    "shapeFunctions.editCategoricalMapping":
      "Kategoriale Zuordnung bearbeiten",
    "shapeFunctions.editChartMapping":
      "Diagrammzuordnung bearbeiten",
    "shapeFunctions.chartMappingButton": "Diagrammzuordnung",
    "shapeFunctions.enlargeChart": "Diagramm vergrößern",
    "shapeFunctions.submitFeature": "Senden",
    "shapeFunctions.dataDistribution": "Datenverteilung",
    "shapeFunctions.showDistribution": "Verteilung anzeigen",
    "shapeFunctions.hideDistribution": "Verteilung ausblenden",
    "shapeFunctions.countAxisLabel": "Anzahl",
    "shapeFunctions.distributionTotalCount":
      "Verwendete Datenpunkte ohne fehlende Werte: {count}",
    "shapeFunctions.distributionAnsweredCount":
      "Beantwortet / ohne fehlende Werte: {count}",
    "shapeFunctions.distributionMissingCount": "Fehlend: {count}",
    "shapeFunctions.distributionRepresentedCount":
      "Insgesamt dargestellte Zeilen: {count}",
    "shapeFunctions.distributionEmpty": "Keine Verteilungsdaten verfügbar.",
    "shapeFunctions.openFeatureDetails": "Feature-Details öffnen",
    "shapeFunctions.detailsButton": "Details",
    "shapeFunctions.detailsDrawerTitle": "Feature-Details",
    "shapeFunctions.detailsSummary":
      "Zeigt, wie dieses Feature im Datensatz konstruiert wurde.",
    "shapeFunctions.detailsUnavailable":
      "Für dieses Feature sind keine Metadaten zur Herkunft verfügbar.",
    "shapeFunctions.detailsCategory": "Kategorie",
    "shapeFunctions.detailsConstruction": "Konstruktion",
    "shapeFunctions.detailsSourceVariables": "Quellvariablen",
    "shapeFunctions.detailsSourceQuestions": "Quellvariablen",
    "shapeFunctions.detailsAnswerScale": "Antwortskala",
    "shapeFunctions.detailsScaleItems": "Quellfragen",
    "shapeFunctions.detailsScaleItemsSummary":
      "Dies ist ein imputierter Skalenwert, der aus den folgenden Aussagen gebildet wird.",
    "shapeFunctions.detailsTransformation": "Transformation",
    "shapeFunctions.detailsMissingHandling": "Fehlende Werte",
    "shapeFunctions.detailsRationale":
      "Warum dieses Feature ausgewählt wurde",
    "shapeFunctions.detailsSourceCount": "{count} Quellfragen",
    "shapeFunctions.detailsRawSource": "Direkter Quellwert",
    "shapeFunctions.detailsItemMean": "Itemmittelwert",
    "shapeFunctions.detailsIqbScale": "IQB-Skalenwert",
    "shapeFunctions.detailsUnknownConstruction": "Abgeleitetes Feature",
    "shapeFunctions.detailAnswerOption.agreement.1": "stimmt gar nicht",
    "shapeFunctions.detailAnswerOption.agreement.2": "stimmt eher nicht",
    "shapeFunctions.detailAnswerOption.agreement.3": "stimmt eher",
    "shapeFunctions.detailAnswerOption.agreement.4": "stimmt genau",
    "shapeFunctions.detailQuestion.Sdruckd01a":
      "Im Deutschunterricht wird viel verlangt.",
    "shapeFunctions.detailQuestion.Sdruckd01b":
      "Im Deutschunterricht gibt es oft unangekündigte Tests.",
    "shapeFunctions.detailQuestion.Sdruckd01c":
      "Wenn man zu oft krank ist, verpasst man im Deutschunterricht schnell den Anschluss.",
    "shapeFunctions.detailQuestion.Sdruckd01d":
      "Wenn man am Wochenende nicht lernt, kommt man im Deutschunterricht kaum mit.",
    "shapeFunctions.detailQuestion.Sdruckd01e":
      "Im Deutschunterricht kommen wir mit den Hausaufgaben kaum nach.",
    "shapeFunctions.detailQuestion.Ssoe01a":
      "Ich bin in meiner Klasse beliebt.",
    "shapeFunctions.detailQuestion.Ssoe01b":
      "Die anderen Kinder in meiner Klasse kommen mit ihren Problemen zu mir.",
    "shapeFunctions.detailQuestion.Ssoe01c":
      "Ich habe wirklich gute Freund:innen in meiner Klasse.",
    "shapeFunctions.detailQuestion.Ssoe01d":
      "Die anderen Kinder in meiner Klasse mögen mich.",
    "shapeFunctions.detailQuestion.Sind01a": "Deutsch ist wichtig.",
    "shapeFunctions.detailQuestion.Sind01b": "Deutsch macht Spaß.",
    "shapeFunctions.detailQuestion.Sind01c": "Ich interessiere mich für Deutsch.",
    "shapeFunctions.detailQuestion.Sind01d":
      "Deutsch ist eines meiner Lieblingsfächer.",
    "shapeFunctions.detailQuestion.Sword01a":
      "Ich mache mir Sorgen, dass die Zeit in einem Deutschtest nicht reicht.",
    "shapeFunctions.detailQuestion.Sword01b":
      "Ich mache mir Sorgen, dass mich die Lehrkraft im Deutschunterricht aufruft.",
    "shapeFunctions.detailQuestion.Sword01c":
      "Ich mache mir Sorgen, dass ich die Deutschhausaufgaben nicht lösen kann.",
    "shapeFunctions.detailQuestion.Sword01d":
      "Ich mache mir Sorgen, dass ich im Deutschunterricht etwas nicht verstehe.",
    "shapeFunctions.detailQuestion.Semod01a":
      "Ich bin aufgeregt, wenn die Zeit in einem Deutschtest nicht reicht.",
    "shapeFunctions.detailQuestion.Semod01b":
      "Ich bin aufgeregt, wenn mich die Lehrkraft im Deutschunterricht aufruft.",
    "shapeFunctions.detailQuestion.Semod01c":
      "Ich bin aufgeregt, wenn ich die Deutschhausaufgaben nicht lösen kann.",
    "shapeFunctions.detailQuestion.Semod01d":
      "Ich bin aufgeregt, wenn ich im Deutschunterricht etwas nicht verstehe.",
    "shapeFunctions.detailQuestion.Sumz21a":
      "Ich kann mit einer Tastatur schreiben.",
    "shapeFunctions.detailQuestion.Sumz21b": "Ich kann mit einer Maus klicken.",
    "shapeFunctions.detailQuestion.Sumz21c":
      "Ich kann Aufgaben auf einem Tablet bearbeiten.",
    "shapeFunctions.detailQuestion.Sumz21d":
      "Ich kann Aufgaben an einem Computer bearbeiten.",
    "shapeFunctions.detailQuestion.Sumz21e":
      "Ich kann mit einer Suchmaschine etwas finden.",
    "shapeFunctions.detailQuestion.Sulkd01a":
      "Meine Deutschlehrkraft kümmert sich um mich.",
    "shapeFunctions.detailQuestion.Sulkd01b":
      "Meine Deutschlehrkraft macht mir Mut.",
    "shapeFunctions.detailQuestion.Sulkd01c":
      "Meine Deutschlehrkraft gibt mir Tipps.",
    "shapeFunctions.detailQuestion.Sulkd01d":
      "Meine Deutschlehrkraft sagt mir, was ich schon gut kann und was ich noch lernen muss.",
    "shapeFunctions.detailQuestion.Sulkd01e":
      "Meine Deutschlehrkraft nimmt sich Zeit für mich.",
    "shapeFunctions.detailQuestion.Sulkd01f":
      "Meine Deutschlehrkraft lobt mich.",
    "shapeFunctions.detailQuestion.Sulkd01g":
      "Meine Deutschlehrkraft gibt mir Hinweise zur Verbesserung.",

    "predictionComparison.title": "Vorhersagevergleich",
    "predictionComparison.empty":
      "Bearbeite Shape-Funktionen und wende Änderungen an, um den Vorhersagevergleich zu sehen.",
    "predictionComparison.subtitleWithEdits":
      "Vergleiche Vorhersagen zwischen dem ursprünglichen IGANN-Modell und deiner bearbeiteten Version.",
    "predictionComparison.subtitleWithoutEdits":
      "Bearbeite Shape-Funktionen, um zu sehen, wie sich Vorhersagen ändern.",
    "predictionComparison.customizeColors":
      "Diagrammfarben anpassen",
    "predictionComparison.colorsButton": "Farben",
    "predictionComparison.chartColors": "Diagrammfarben",
    "predictionComparison.originalModelDots":
      "Punkte des Originalmodells",
    "predictionComparison.editedModelDots":
      "Punkte des bearbeiteten Modells",
    "predictionComparison.perfectPredictionLine":
      "Linie perfekter Vorhersage",
    "predictionComparison.persistHint":
      "Melde dich an, um Farben sitzungsübergreifend zu speichern.",
    "predictionComparison.originalTrace": "IGANN (Original)",
    "predictionComparison.editedTrace":
      "IGANN Interaktiv (Bearbeitet)",
    "predictionComparison.perfectPrediction":
      "Perfekte Vorhersage",
    "predictionComparison.plotTitle": "Vorhersagen vs. Ist-Werte",
    "predictionComparison.plotTitleWithEdits":
      "Vorhersagen vs. Ist-Werte (Original vs. Bearbeitet)",
    "predictionComparison.actualBikeRentals":
      "Tatsächliche Fahrradvermietungen",
    "predictionComparison.predictedBikeRentals":
      "Vorhergesagte Fahrradvermietungen",
    "predictionComparison.originalModel": "IGANN (Original)",
    "predictionComparison.editedModel":
      "IGANN Interaktiv (Bearbeitet)",
    "predictionComparison.summary": "Zusammenfassung:",
    "predictionComparison.improved": "verbessert",
    "predictionComparison.worsened": "verschlechtert",

    "combined.title": "Kombinierte Ergebnisse - Alle Benutzer",
    "combined.refresh": "Aktualisieren",
    "combined.loading": "Kombinierte Ergebnisse werden geladen...",
    "combined.summaryTitle": "Kombinierte Analyse von {count} Benutzern",
    "combined.summaryDescription":
      "Diese Seite zeigt den aggregierten Effekt aller Benutzerbearbeitungen auf das GAM-Modell. Der Offset jedes Punkts wird über alle Benutzer gemittelt, die ihn bearbeitet haben.",
    "combined.participatingUsers":
      "Teilnehmende Benutzer ({count})",
    "combined.noUsers": "Noch keine Benutzer.",
    "combined.originalModel": "Originalmodell",
    "combined.combinedUserEdits":
      "Kombinierte Benutzerbearbeitungen",
    "combined.rmse": "RMSE",
    "combined.mae": "MAE",
    "combined.r2": "R2",
    "combined.scatterTitle":
      "Vorhergesagt vs. Original (Streudiagramm)",
    "combined.scatterDescription":
      "Punkte näher an der Diagonalen zeigen bessere Vorhersagen. Vergleiche, wie sich kombinierte Benutzerbearbeitungen auf die Genauigkeit auswirken.",
    "combined.formTitle": "Mit kombinierten Bearbeitungen vorhersagen",
    "combined.formDescription":
      "Erstelle eine Vorhersage mit dem Modell, auf das alle kombinierten Benutzerbearbeitungen angewendet wurden. Das Ergebnis spiegelt die aggregierten Änderungen der Shape-Funktionen aller Benutzer wider.",
    "combined.predictButton":
      "{target} vorhersagen (kombiniertes Modell)",
    "combined.predictedResult":
      "Vorhergesagtes {target} (kombinierte Bearbeitungen)",
    "combined.predictionFailed": "Vorhersage fehlgeschlagen",
    "combined.shapeFunctionsNone":
      "Keine Shape-Funktionsdaten verfügbar.",
    "combined.shapeFunctionsTitle":
      "Shape-Funktionen: Original vs. kombinierte Bearbeitungen",
    "combined.weightingToggleTitle":
      "Umschalten, ob die kombinierte Linie eine sicherheitsgewichtete Mittelung oder einen einfachen Mittelwert verwendet",
    "combined.weightingOn": "Gewichtung: An",
    "combined.weightingOff": "Gewichtung: Aus",
    "combined.showUserOverlay": "Benutzer-Overlay anzeigen",
    "combined.loadOverlayFailed":
      "Benutzer-Overlays konnten nicht geladen werden. Bitte erneut versuchen.",
    "combined.updateWeightingFailed":
      "Die Gewichtung konnte nicht geändert werden. Bitte erneut versuchen.",
    "combined.hideUserOverlay": "Benutzer-Overlay ausblenden",
    "combined.legend": "Legende:",
    "combined.original": "Original",
    "combined.combined": "Kombiniert",
    "combined.comparison": "Vergleichsdatensatz",
    "combined.noEdits":
      "Es wurden noch keine Benutzerbearbeitungen vorgenommen. Die Diagramme unten zeigen die ursprünglichen Shape-Funktionen.",
    "combined.modified": "Geändert",
    "combined.effect": "Effekt",
    "combined.editLogsUnavailable":
      "Bearbeitungsprotokolle sind vorübergehend nicht verfügbar: {error}",
    "combined.editLogsTitle": "Bearbeitungsprotokolle",
    "combined.editLogsDescription":
      "Detailliertes Protokoll aller Benutzerbearbeitungen, nach Feature gruppiert. Es zeigt, wer bearbeitet hat, die selbst angegebene Sicherheit (1-10), den rohen Eingabewert und das gewichtete Ergebnis in der kombinierten Ansicht.",
    "combined.submissionSingular": "Einreichung",
    "combined.submissionPlural": "Einreichungen",
    "combined.userSingular": "Benutzer",
    "combined.userPlural": "Benutzer",
    "combined.by": "von",
    "combined.user": "Benutzer",
    "combined.submitted": "Eingereicht",
    "combined.confidence": "Sicherheit",
    "combined.points": "Punkte",
    "combined.rawTotal": "Rohe Summe",
    "combined.weightedTotal": "Gewichtete Summe",
    "combined.message": "Nachricht",
    "combined.xSummary": "X-Übersicht",
    "combined.deleteSubmission": "Einreichung löschen",
    "combined.xValue": "X-Wert",
    "combined.rawInput": "Roheingabe",
    "combined.weighted": "Gewichtet",
    "combined.originalLinePreview": "Originale Linie",
    "combined.weightedLinePreview": "Gewichtete bearbeitete Linie",
    "combined.noSubmissions":
      "Keine Einreichungen für dieses Feature verfügbar.",
    "combined.deleteDisabled":
      "Destruktive Aktionen sind in dieser Demo deaktiviert.",
    "combined.deleteReasonRequired":
      "Bitte gib einen Grund für das Löschen dieser Bearbeitung an.",
    "combined.noDeletionTarget":
      "Kein Löschziel für diese Einreichung verfügbar",
    "combined.deleteFailed":
      "Einreichung konnte nicht gelöscht werden",
    "combined.loadEditLogsFailed":
      "Bearbeitungsprotokolle konnten nicht geladen werden",
    "combined.loadDataFailed": "Daten konnten nicht geladen werden",
    "combined.deleteModalTitle": "Einreichung löschen",
    "combined.deleteModalDescription":
      "Du bist dabei, eine eingereichte Kurvenbearbeitung von {user} zu löschen.",
    "combined.feature": "Feature",
    "combined.whyRemoved":
      "Warum wird diese Einreichung entfernt?",
    "combined.deleteReasonPlaceholder":
      "Gib einen Grund für das Entfernen dieser Einreichung an...",
    "combined.deleting": "Löschen...",

    "superadmin.error.loadUsers": "Benutzer konnten nicht geladen werden",
    "superadmin.error.usernamePasswordRequired":
      "Benutzername und Passwort sind erforderlich",
    "superadmin.error.usernamePasswordProfessionRequired":
      "Benutzername, Passwort und Beruf sind erforderlich",
    "superadmin.error.createUser":
      "Benutzer konnte nicht erstellt werden",
    "superadmin.error.createInvite":
      "Einladung konnte nicht erstellt werden",
    "superadmin.error.resetDatabase":
      "Datenbank konnte nicht zurückgesetzt werden",
    "superadmin.error.exportModel":
      "Modell konnte nicht exportiert werden",
    "superadmin.error.importModel":
      "Modell konnte nicht importiert werden",
    "superadmin.compareUploadError":
      "Vergleichsdatensatz konnte nicht hochgeladen werden",
    "superadmin.compareLoadError":
      "Vergleichsdatensatz konnte nicht geladen werden",
    "superadmin.title": "Superadmin-Übersicht",
    "superadmin.openCombined": "Kombinierte Ergebnisse öffnen",
    "superadmin.openModelCompare": "Modellexporte vergleichen",
    "superadmin.compareDatasets": "Datensätze vergleichen",
    "superadmin.compareDatasetsDescription":
      "Lade einen zweiten Datensatz hoch, trainiere ihn mit der Estimator-Anzahl des primären Modells und lege seine Shape-Funktionen über die interaktiven Diagramme.",
    "superadmin.chartDisplayTitle": "Diagrammanzeige",
    "superadmin.chartDisplayDescription":
      "Steuere globale Anzeigeoptionen fuer Shape-Funktionsdiagramme und Vorhersageergebnisse.",
    "superadmin.showMissingBars": "Fehlend-Balken anzeigen",
    "superadmin.hideMissingBars": "Fehlend-Balken ausblenden",
    "superadmin.showCompetenceLevels": "Kompetenztabelle anzeigen",
    "superadmin.hideCompetenceLevels": "Kompetenztabelle ausblenden",
    "superadmin.predictionComparisonVisibilityTitle":
      "Sichtbarkeit des Vorhersagevergleichs",
    "superadmin.predictionComparisonVisibilityDescription":
      "Lege fest, wer den Vorhersagevergleich auf der Hauptseite sehen kann. Beide Ansichten sind standardmäßig ausgeblendet.",
    "superadmin.showPredictionComparisonSelf": "Für mich anzeigen",
    "superadmin.hidePredictionComparisonSelf": "Für mich ausblenden",
    "superadmin.showPredictionComparisonUsers": "Für andere anzeigen",
    "superadmin.hidePredictionComparisonUsers": "Für andere ausblenden",
    "superadmin.updatingChartDisplay":
      "Diagrammanzeige wird aktualisiert...",
    "superadmin.error.updateChartDisplaySettings":
      "Diagrammanzeige konnte nicht aktualisiert werden",
    "superadmin.compareModalDescription":
      "Der Vergleichsdatensatz muss dieselbe Zielspalte und dieselben ausgewählten Feature-Spalten wie der primäre Datensatz verwenden.",
    "superadmin.uploadAndSelectComparison":
      "Hochladen und Features wählen",
    "superadmin.trainComparison": "Vergleichsmodell trainieren",
    "superadmin.primaryEstimators": "Primäre Estimatoren",
    "superadmin.comparisonDataset": "Vergleichsdatensatz",
    "superadmin.noComparisonDataset": "Kein Vergleichsdatensatz geladen",
    "superadmin.comparisonTrained": "Vergleich trainiert",
    "superadmin.loadComparisonDataset": "Vergleichsdatensatz laden",
    "superadmin.sensitivityTitle": "Sensitivitätsprüfung der Vorverarbeitung",
    "superadmin.sensitivityDescription":
      "Lade die ursprüngliche vollständige und die 5-%-IQB-CSV mit den passenden unbearbeiteten IGANN-Exporten hoch. Behalte die ursprünglichen CSV-Dateinamen bei. Beide Modelle werden mit Medianen aus den jeweiligen Trainingszeilen neu trainiert und auf ihren ursprünglichen 20-%-Testsätzen bewertet. Das aktive Modell bleibt unverändert.",
    "superadmin.sensitivityFullCsv": "Vollständige CSV",
    "superadmin.sensitivityFullModel": "Ursprüngliches vollständiges Modell (JSON)",
    "superadmin.sensitivityReducedCsv": "5-%-CSV",
    "superadmin.sensitivityReducedModel": "Ursprüngliches 5-%-Modell (JSON)",
    "superadmin.sensitivityRun": "Sensitivitätsprüfung starten",
    "superadmin.sensitivityRunning": "Beide Modelle werden neu trainiert...",
    "superadmin.sensitivityError": "Sensitivitätsprüfung fehlgeschlagen",
    "superadmin.sensitivityMethodNote":
      "Der Modellexport wird aus gespeicherten Shape-Funktionen bewertet; dies kann native Vorhersagen annähern. Die kontrollierte Differenz vergleicht zwei native Neutrainings mit gleichen Einstellungen und Testzeilen; nur die Medianberechnung ändert sich. Niedrigere MAE/RMSE und ein höheres R² sind besser.",
    "superadmin.sensitivityFullResult": "Vollständiges Modell",
    "superadmin.sensitivityReducedResult": "5-%-Modell",
    "superadmin.sensitivityRowCounts":
      "{usable} verwendbare Zeilen · {train} Training · {test} Test",
    "superadmin.sensitivityMaePercent":
      "MAE-Änderung gegenüber dem kontrollierten ursprünglichen Neutraining: {percent} %",
    "superadmin.sensitivityModel": "Modell",
    "superadmin.sensitivityExported": "Ursprünglicher Export (rekonstruiert)",
    "superadmin.sensitivityRefitOld": "Neutraining mit ursprünglicher Vorverarbeitung",
    "superadmin.sensitivityCorrected": "Neutraining mit Trainingsmedianen",
    "superadmin.sensitivityDelta": "Änderung gegenüber ursprünglichem Neutraining",
    "superadmin.sensitivityMedians": "Durch die Korrektur veränderte Mediane",
    "superadmin.sensitivityNoMedianChanges": "Keine numerischen Ersatzmediane haben sich verändert.",
    "superadmin.createUser": "Benutzer erstellen",
    "superadmin.professionPlaceholder": "Beruf eingeben",
    "superadmin.inviteLink": "Einladungslink",
    "superadmin.generateInvite": "Einladung erzeugen",
    "superadmin.inviteCopied":
      "Einladungslink wurde in die Zwischenablage kopiert.",
    "superadmin.expires": "Läuft ab",
    "superadmin.inviteDescription":
      "Einen Einladungslink erzeugen, um neue Registrierungen zu ermöglichen.",
    "superadmin.systemReset": "System zurücksetzen",
    "superadmin.systemResetDescription":
      "Entfernt dauerhaft alle Benutzer, alle gespeicherten Bearbeitungen, hochgeladene CSV-Dateien und den extrahierten Feature-Zustand der aktuellen Backend-Umgebung.",
    "superadmin.resetDatabase": "Datenbank zurücksetzen",
    "superadmin.creating": "Wird erstellt...",
    "superadmin.users": "Benutzer",
    "superadmin.loadingUsers": "Benutzer werden geladen...",
    "superadmin.noUsers": "Noch keine Benutzer.",
    "superadmin.created": "Erstellt",
    "superadmin.badge": "Superadmin",
    "superadmin.resetConfirmTitle": "Datenbank zurücksetzen?",
    "superadmin.resetConfirmBody":
      "Dadurch werden Benutzer, Bearbeitungen, hochgeladene CSV-Dateien und extrahierter Feature-Zustand dauerhaft gelöscht. Diese Aktion kann nicht rückgängig gemacht werden.",
    "superadmin.resetting": "Zurücksetzen läuft...",
    "superadmin.resetEverything": "Ja, alles zurücksetzen",
    "superadmin.modelTransferTitle": "Modelltransfer",
    "superadmin.modelTransferDescription":
      "Exportiere das aktive trainierte Modell als JSON oder importiere ein zuvor exportiertes Modellartefakt.",
    "superadmin.exporting": "Export läuft...",
    "superadmin.exportModel": "Modell exportieren",
    "superadmin.importing": "Import läuft...",
    "superadmin.importModel": "Modell importieren",
    "superadmin.modelTransferWarning":
      "Beim Importieren eines Modells wird das aktuelle Basismodell ersetzt und gespeicherte Benutzerbearbeitungen, die mit dem vorherigen Modell verknüpft sind, werden gelöscht.",
    "superadmin.exportSuccess": "Modell wurde als {filename} exportiert.",
    "superadmin.exportSuccessWithEdits":
      "Modell und gespeicherte Shape-Function-Bearbeitungen wurden als {filename} exportiert.",
    "superadmin.importSuccess":
      "Modell wurde erfolgreich importiert.",
    "superadmin.importSuccessWithEdits":
      "Modell und gespeicherte Shape-Function-Bearbeitungen wurden erfolgreich für {users} Benutzer mit {submissions} Einreichungen importiert.",
    "superadmin.exportConfirmTitle": "Modellartefakt exportieren?",
    "superadmin.exportConfirmBody":
      "Wähle, ob das exportierte Modellartefakt auch gespeicherte Shape-Function-Bearbeitungen des aktuellen Modells enthalten soll. Der Export enthält außerdem die zurückgehaltenen Testzeilen und Zielwerte für eine spätere Bewertung.",
    "superadmin.exportIncludeEdits":
      "Gespeicherte Shape-Function-Bearbeitungen einschließen",
    "superadmin.exportConfirmAction": "Artefakt exportieren",
    "modelCompare.title": "Vergleich exportierter Modelle",
    "modelCompare.subtitle":
      "Zwei exportierte Modell-JSON-Dateien hochladen und ihre Basis- und bearbeiteten Feature-Shapes vergleichen.",
    "modelCompare.previewOnly":
      "Nur Vorschau - das Live-Modell bleibt unverändert",
    "modelCompare.leftArtifact": "Artefakt mit vollständigen Daten",
    "modelCompare.rightArtifact": "Artefakt mit reduzierten Daten",
    "modelCompare.leftArtifactSelections": "Auswahl des Artefakts mit vollständigen Daten",
    "modelCompare.rightArtifactSelections": "Auswahl des Artefakts mit reduzierten Daten",
    "modelCompare.noFileChosen": "Noch keine Datei ausgewählt",
    "modelCompare.compareArtifacts": "Artefakte vergleichen",
    "modelCompare.instructions":
      "Beide Uploads müssen exportierte Modell-JSON-Dateien mit derselben Zielspalte und derselben Feature-Auswahl sein.",
    "modelCompare.preparing": "Vergleich wird vorbereitet...",
    "modelCompare.prepareError":
      "Modellvergleich konnte nicht vorbereitet werden",
    "modelCompare.previewError":
      "Vergleichsvorschau konnte nicht neu berechnet werden",
    "modelCompare.chooseBothFiles":
      "Bitte wähle beide exportierten JSON-Dateien aus, bevor du sie vergleichst.",
    "modelCompare.overlapTitle": "CSV-Überschneidung mit den Artefakt-Testsätzen",
    "modelCompare.overlapDescription":
      "Die Quell-CSV des reduzierten Modells hochladen, um dessen tatsächliche Trainingszeilen mit dem Original-Testsatz zu vergleichen. Für eine exakte Prüfung ohne gemeinsame Quell-IDs zusätzlich die ältere vollständige Quell-CSV hochladen. Der bisherige CSV-Zeilenvergleich steht darunter separat.",
    "modelCompare.overlapCsvLabel": "CSV mit reduzierten Daten",
    "modelCompare.overlapSourceCsvLabel": "Ältere vollständige Quell-CSV (optional)",
    "modelCompare.overlapCheck": "Überschneidung prüfen",
    "modelCompare.overlapChecking": "Überschneidung wird geprüft...",
    "modelCompare.overlapError": "CSV-Überschneidung konnte nicht geprüft werden",
    "modelCompare.overlapUnavailable":
      "Das Originalartefakt enthält keine zurückgehaltenen Testzeilen. Exportiere ein Modell mit Testdaten, um diese Prüfung auszuführen.",
    "modelCompare.overlapEditedUnavailable":
      "Das bearbeitete Artefakt enthält keine zurückgehaltenen Testzeilen zum Abziehen. Die CSV wird nur mit dem Original-Testsatz verglichen.",
    "modelCompare.overlapResult":
      "{matches} von {remaining} verbleibenden CSV-Zeilen stimmen mit dem Original-Testsatz überein ({testRows} Testzeilen).",
    "modelCompare.overlapDirectResult":
      "{matches} von {remaining} CSV-Zeilen stimmen mit dem Original-Testsatz überein ({testRows} Testzeilen).",
    "modelCompare.overlapSubtraction":
      "CSV-Zeilen: {csvRows}. Wegen Übereinstimmung mit dem Testsatz des bearbeiteten Artefakts ({editedTestRows} Testzeilen) entfernt: {removed}. Verbleibend: {remaining}.",
    "modelCompare.overlapBeforeSubtraction":
      "Vor dem Abzug stimmten {matches} CSV-Zeilen mit dem Original-Testsatz überein.",
    "modelCompare.overlapWithTarget":
      "Der Vergleich verwendete alle Modell-Features und die Zielspalte.",
    "modelCompare.overlapFeaturesOnly":
      "Die CSV enthält keine Original-Zielspalte; der Vergleich verwendete nur Modell-Features.",
    "modelCompare.overlapMatchedRows": "Verbleibende Zeilen mit Übereinstimmung im Original-Testsatz (nach der CSV-Kopfzeile nummeriert):",
    "modelCompare.overlapDirectMatchedRows": "Zeilen mit Übereinstimmung im Original-Testsatz (nach der CSV-Kopfzeile nummeriert):",
    "modelCompare.overlapRemovedRows": "Entfernte CSV-Zeilen mit Übereinstimmung im bearbeiteten Testsatz ({count})",
    "modelCompare.overlapRowDetail":
      "CSV-Zeile {csvRow} stimmt mit Testzeile {testRows} überein",
    "modelCompare.overlapSampleIds": "Stichproben-ID: {ids}",
    "modelCompare.overlapShowingFirst": "Die ersten {count} übereinstimmenden CSV-Zeilen werden angezeigt.",
    "modelCompare.trainingOverlapTitle": "Trainingszeilen des reduzierten Modells im Original-Testsatz",
    "modelCompare.trainingOverlapExact":
      "{matches} von {training} Trainingsbeobachtungen des reduzierten Modells kommen im zurückgehaltenen Testsatz des Originalmodells mit {test} Zeilen vor.",
    "modelCompare.trainingOverlapSourceVerified":
      "Durch Zuordnung vollständiger Zeilen der reduzierten CSV zur älteren vollständigen Quell-CSV und Prüfung beider Artefakt-Aufteilungen verifiziert.",
    "modelCompare.trainingOverlapIdsVerified":
      "Anhand gemeinsamer Quell-IDs in beiden Artefakten verifiziert.",
    "modelCompare.trainingOverlapTargetOnly":
      "{matches} von {training} reduzierten Trainingszeilen haben eindeutige Zielwerte, die auch im Original-Testsatz mit {test} Zeilen vorkommen.",
    "modelCompare.trainingOverlapNeedsSource":
      "Gleiche Zielwerte beweisen noch keine identischen Beobachtungen. Lade die ältere vollständige Quell-CSV hoch, um die genaue Überschneidung zu verifizieren.",
    "modelCompare.trainingOverlapUnavailable":
      "Die Überschneidung von Training und Test konnte nicht verifiziert werden: {reason}",
    "modelCompare.trainingOverlapUsableRows":
      "Verwertbare Zeilen der reduzierten CSV nach Ausschluss fehlender Zielwerte: {count}.",
    "modelCompare.overlapDescriptiveTitle":
      "Weitere CSV-Zeilenübereinstimmungen (nicht die Trainingsdaten-Überschneidung)",
    "modelCompare.exportedAt": "Exportiert am",
    "modelCompare.dataset": "Datensatz",
    "modelCompare.features": "Features",
    "modelCompare.editExport": "Edit-Export",
    "modelCompare.selectedSubmissions": "Ausgewählte Einreichungen",
    "modelCompare.selectionHint":
      "Importierte Benutzereinreichungen auswählen oder abwählen, um ihren Effekt in der Vorschau zu sehen.",
    "modelCompare.selected": "ausgewählt",
    "modelCompare.noImportedEdits":
      "Dieses Artefakt enthält keine importierten Benutzerbearbeitungen.",
    "modelCompare.noImportedEditsForFeature":
      "Dieses Artefakt enthält für dieses Feature keine importierten Benutzerbearbeitungen.",
    "modelCompare.confidence": "Sicherheit",
    "modelCompare.points": "Punkte",
    "modelCompare.xSummary": "X-Übersicht",
    "modelCompare.chartControls": "Diagrammsteuerung",
    "modelCompare.chartControlsDescription":
      "Zwischen sicherheitsgewichteter und gleich gewichteter Vorschau wechseln, Features suchen und zu einem Diagramm springen.",
    "modelCompare.confidenceOn": "Sicherheitsgewichtung an",
    "modelCompare.confidenceOff": "Sicherheitsgewichtung aus",
    "modelCompare.showAlternateConfidenceAggregation":
      "Beide Sicherheitsgewichtungen vergleichen",
    "modelCompare.hideAlternateConfidenceAggregation":
      "Nur ausgewählte Gewichtung anzeigen",
    "modelCompare.featureMetricsOriginalTestSet":
      "Originalen Testsatz für MAE, RMSE und R² je Feature verwenden",
    "modelCompare.featureMetricsUniformGridDescription":
      "Wenn ausgeschaltet, verwenden numerische Metriken 100 gleichmäßig verteilte Punkte im gemeinsamen Kurvenbereich.",
    "modelCompare.featureMetricsOriginalTestSetUnavailable":
      "Erfordert zurückgehaltene Feature-Zeilen im Originalartefakt.",
    "modelCompare.confidenceDescription":
      "Wenn aus, werden ausgewählte Einreichungen gleich gewichtet statt mit ihrer gespeicherten Sicherheit.",
    "modelCompare.showBaseTraces": "Basislinien und Balken anzeigen",
    "modelCompare.hideBaseTraces": "Basislinien und Balken ausblenden",
    "modelCompare.centerCharts": "Kurven zentrieren",
    "modelCompare.restoreChartPositions": "Kurvenpositionen wiederherstellen",
    "modelCompare.showLineDots": "Punkte auf Linien anzeigen",
    "modelCompare.hideLineDots": "Punkte auf Linien ausblenden",
    "modelCompare.makeChartsTaller": "Diagramme höher anzeigen",
    "modelCompare.restoreChartHeight": "Diagrammhöhe zurücksetzen",
    "modelCompare.baseTracesDescription":
      "Die ursprünglichen Basistraces für beide verglichenen Artefakte ein- oder ausblenden.",
    "modelCompare.showIndividualEdits": "Einzelne Bearbeitungen anzeigen",
    "modelCompare.showAggregatedEdits": "Zusammengefasste Bearbeitungen anzeigen",
    "modelCompare.enableChartExpertSelection": "Expert:innen in Feature-Diagrammen auswählen",
    "modelCompare.disableChartExpertSelection": "Expert:innen-Auswahl in Feature-Diagrammen beenden",
    "modelCompare.chartExpertSelectionHint":
      "Wähle Expert:innen für die Formfunktions- und MAE-Änderungsdiagramme dieses Features. Andere Ergebnisse bleiben unverändert.",
    "modelCompare.showIndividualMae": "Individuelle MAE-Fehler anzeigen",
    "modelCompare.hideIndividualMae": "Individuelle MAE-Fehler ausblenden",
    "modelCompare.individualMaeErrors": "Individuelle MAE-Fehler der Teilnehmenden",
    "modelCompare.showIndividualMaeChangeChart": "MAE-Änderungsdiagramm der Expert:innen anzeigen",
    "modelCompare.hideIndividualMaeChangeChart": "MAE-Änderungsdiagramm der Expert:innen ausblenden",
    "modelCompare.showConfidenceRatedMae": "Sicherheitsgewichteten Expert:innen-MAE anzeigen",
    "modelCompare.hideConfidenceRatedMae": "Sicherheitsgewichteten Expert:innen-MAE ausblenden",
    "modelCompare.expertMaeChange": "MAE-Änderung der Expert:innen",
    "modelCompare.expertMaeChangeTitle": "Änderung des Formfunktions-MAE je Expert:in",
    "modelCompare.expertMaeChangeAxis": "MAE-Δ zur reduzierten Daten-Baseline",
    "modelCompare.allFeatureMaeChange":
      "Zusammengefasste MAE-Änderung über alle Features",
    "modelCompare.allFeatureMaeChangeDescription":
      "Jeder Balken zeigt ein Feature nach allen exportierten Bearbeitungen im Vergleich zu den unbearbeiteten Modellen. Beide Sicherheitsmodi sind unabhängig von der Expert:innen-Auswahl in den Feature-Diagrammen. Im kombinierten Diagramm kennzeichnen gemusterte Balken die Sicherheitsgewichtung.",
    "modelCompare.expertAverageConfidence": "Durchschnittliche Sicherheit je Expert:in",
    "modelCompare.expertAverageConfidenceDescription":
      "Durchschnitt der gespeicherten Sicherheitsbewertungen über alle exportierten Feature-Einreichungen je Expert:in. Die Balken sind absteigend sortiert und unabhängig von der Auswahl in Feature-Diagrammen.",
    "modelCompare.expertAverageConfidenceAxis": "Durchschnittliche Sicherheitsbewertung (1–10)",
    "modelCompare.expertAverageConfidenceRatings": "Bewertete Einreichungen",
    "modelCompare.noExpertConfidenceRatings": "Keine gespeicherten Sicherheitsbewertungen von Expert:innen verfügbar.",
    "modelCompare.allFeatureMaeChangeWithoutConfidence":
      "Ohne Sicherheitsbewertung",
    "modelCompare.allFeatureMaeChangeWithConfidence":
      "Mit Sicherheitsbewertung",
    "modelCompare.showCombinedMaeChart": "Kombiniertes Diagramm anzeigen",
    "modelCompare.showSeparateMaeCharts": "Getrennte Diagramme anzeigen",
    "modelCompare.showConfidenceInLegend": "Sicherheitsbewertung in Legende anzeigen",
    "modelCompare.hideConfidenceInLegend": "Sicherheitsbewertung in Legende ausblenden",
    "modelCompare.individualEditsDescription":
      "Eine durchgezogene Kurve für jede ausgewählte Bearbeitung statt der zusammengefassten Kurve anzeigen; Farben kennzeichnen die Expert:innen.",
    "modelCompare.legendRightOfChart": "Legende nach rechts verschieben",
    "modelCompare.legendAboveChart": "Legende über dem Diagramm anzeigen",
    "modelCompare.legendPositionDescription":
      "Die Legende jedes Diagramms rechts neben dem Plot oder an ihrer Standardposition darüber anzeigen.",
    "modelCompare.searchFeatures": "Features suchen",
    "modelCompare.jumpToFeature": "Zum Feature springen",
    "modelCompare.featureCharts": "Feature-Diagramme",
    "modelCompare.featureChartsDescription":
      "Gestrichelte Linien zeigen die Basisform jedes Artefakts. Durchgezogene Linien zeigen entweder die zusammengefasste Vorschau oder jede ausgewählte Bearbeitung.",
    "modelCompare.overallShapeFunctionMae": "Gesamt-MAE der Formfunktionen",
    "modelCompare.overallShapeFunctionMaeDescription":
      "Vergleicht die Baseline mit reduzierten Daten und ausgewählte Bearbeitungen mit den Formfunktionen der vollständigen Daten. Niedriger bedeutet größere Nähe.",
    "modelCompare.overallShapeFunctionMaeCoverage":
      "{features} Formfunktionen · {points} Referenzpunkte",
    "modelCompare.pooledShapeFunctionMae": "Zusammengefasste Kurvenpunkte",
    "modelCompare.meanPerFeatureMae": "Gleichgewichteter Feature-Mittelwert",
    "modelCompare.expertCompleteModelMae": "MAE des vollständigen Expert:innenmodells",
    "modelCompare.expertCompleteModelMaeDescription":
      "Jeder Wert vergleicht die ausgewählten Bearbeitungen einer Person über alle verglichenen Formfunktionen mit den Kurven des vollständigen Datensatzes.",
    "modelCompare.expertCompleteModelMaeConfidenceWeighted":
      "Gespeicherte Sicherheitsbewertungen werden auf die Bearbeitungen angewendet.",
    "modelCompare.expertCompleteModelMaeEqualWeighted":
      "Bearbeitungen werden gleich gewichtet; gespeicherte Sicherheitsbewertungen werden nicht angewendet.",
    "modelCompare.expertCompleteModelMaeCoverage":
      "{features} Formfunktionen · {points} Referenzpunkte · {submissions} Einreichungen",
    "modelCompare.overallMaeFeatureSelection":
      "Formfunktionen auswählen ({selected} von {total} ausgewählt)",
    "modelCompare.overallMaeFeatureSelectionDescription":
      "Nur angekreuzte Formfunktionen werden in den Gesamt-MAE-Karten oben berücksichtigt. Mindestens eine muss ausgewählt bleiben.",
    "modelCompare.featureMaeDelta": "MAE-Δ: {delta}",
    "modelCompare.heldOutTestPerformance": "Leistung auf zurückgehaltenen Testdaten",
    "modelCompare.heldOutTestPerformanceDescription":
      "Bewertet das Modell mit vollständigen Daten, das ursprüngliche reduzierte Modell und das bearbeitete reduzierte Modell anhand derselben zurückgehaltenen Zielwerte.",
    "modelCompare.model": "Modell",
    "modelCompare.fullData": "Referenzmodell mit vollständigen Daten",
    "modelCompare.reducedData": "Ursprüngliches reduziertes Modell",
    "modelCompare.editedReducedData": "Bearbeitetes reduziertes Modell",
    "modelCompare.effectOfExpertEdits": "Wirkung der Expertenbearbeitungen",
    "modelCompare.selectedModelComparison": "Vergleich der ausgewählten Modelle",
    "modelCompare.reducedToEdited": "Ursprüngliches reduziertes → bearbeitetes reduziertes Modell",
    "modelCompare.fullToReduced": "Referenzmodell mit vollständigen Daten → ursprüngliches reduziertes Modell",
    "modelCompare.fullToEdited": "Referenzmodell mit vollständigen Daten → bearbeitetes reduziertes Modell",
    "modelCompare.allThreeComparison": "Referenzmodell mit vollständigen Daten → ursprüngliches reduziertes Modell → bearbeitetes reduziertes Modell",
    "modelCompare.reducedMinusFull": "reduziert − vollständig",
    "modelCompare.editedMinusFull": "bearbeitet − vollständig",
    "modelCompare.editedMinusReduced": "bearbeitet − reduziert",
    "modelCompare.allThree": "Alle drei",
    "modelCompare.reducedEdited": "Reduziert vs. bearbeitet",
    "modelCompare.fullReduced": "Vollständig vs. reduziert",
    "modelCompare.fullEdited": "Vollständig vs. bearbeitet",
    "modelCompare.residualComparison": "Tatsächlicher Wert vs. Residuum",
    "modelCompare.residualAxis": "Vorhersage − tatsächlich",
    "modelCompare.zeroResidual": "Residuum null",
    "modelCompare.metric.mae": "MAE",
    "modelCompare.metric.rmse": "RMSE",
    "modelCompare.metric.r2": "R²",
    "modelCompare.metric.mean_error": "Mittlerer Fehler",
    "modelCompare.metric.bias_magnitude": "Betrag des mittleren Vorhersagefehlers",
    "modelCompare.metric.std_dev": "Standardabweichung der Vorhersagen",
    "modelCompare.meanErrorConvention": "Mittlerer Fehler = Mittelwert(Vorhersage − tatsächlich). Negativ bedeutet Unterschätzung, positiv Überschätzung. Die Vergleichskarten verwenden den Betrag des mittleren Vorhersagefehlers.",
    "modelCompare.improved": "Verbessert",
    "modelCompare.worsened": "Verschlechtert",
    "modelCompare.unchanged": "Unverändert",
    "modelCompare.benchmarkRelationships": "Vollständig ↔ reduziert: Lücke durch begrenzte Daten. Vollständig ↔ bearbeitet: verbleibende Lücke. Genauigkeit wird anhand tatsächlicher Zielwerte bewertet.",
    "modelCompare.benchmarkGaps": "Abstand zum Referenzmodell mit vollständigen Daten",
    "modelCompare.observationEffects": "Wirkung je Beobachtung und Unsicherheit",
    "modelCompare.medianErrorChange": "Median der Änderung des absoluten Fehlers",
    "modelCompare.trainingCorrectionOption": "Interzeptkorrektur aus Trainingsdaten anwenden, damit die mittlere Vorhersage des reduzierten Modells erhalten bleibt",
    "modelCompare.manualAdjustmentDisabled": "Die Korrektur aus Trainingsdaten ist aktiv. Deaktiviere sie im Leistungsbereich, um eine manuelle Verschiebung einzugeben.",
    "modelCompare.correctionMetadata": "Interzeptkorrektur aktiviert: {enabled}; Verschiebung: {shift}; Quelle: {source}.",
    "modelCompare.beforeAfterCorrection": "Bearbeitetes Modell vor und nach Interzeptkorrektur",
    "modelCompare.beforeCorrection": "Vor der Korrektur",
    "modelCompare.afterCorrection": "Nach der Korrektur",
    "modelCompare.splitValidation": "Split-Prüfung: Die exportierten Testsätze stimmen überein: {shared}. Abgleichbare Quell-IDs fehlen für mindestens einen Trainingssatz; der Testausschluss ist daher nicht beweisbar. Gleiche Feature-Zeilen in exportierten Trainingsdaten: vollständig {full}, reduziert {reduced}. Treffer können verschiedene Beobachtungen sein.",
    "modelCompare.splitVerified": "Split für {samples} Testzeilen geprüft: Die angegebenen Quell-IDs überschneiden sich mit keinem der beiden Trainingssätze.",
    "modelCompare.preprocessingLeakage": "Hinweis zur Vorverarbeitung: Bei diesen historischen Modellen wurden fehlende Werte vor der Aufteilung ergänzt. Testmerkmale beeinflussten die Vorverarbeitung; Testzielwerte wurden weder zum Modelltraining noch zur Interzeptkorrektur verwendet.",
    "modelCompare.mainTrainingPerformance": "Leistung auf Trainingsdaten (Debug)",
    "modelCompare.mainTrainingPerformanceDescription":
      "Verwendet nur die Zeilen, mit denen das Modell auf der Hauptseite trainiert wurde. Validierungs- und Testzeilen sind ausgeschlossen; diese Kennzahlen dienen der Fehlersuche.",
    "modelCompare.heldOutTestDataSource": "Testsatz",
    "modelCompare.useOriginalArtifactTestSet": "Zurückgehaltener Testsatz des vollständigen Datensatzes",
    "modelCompare.useReducedArtifactTestSet": "Testsatz des reduzierten Artefakts",
    "modelCompare.useMainTrainingData": "Vollständiger Datensatz",
    "modelCompare.evaluateHeldOutTest": "Artefakte bewerten",
    "modelCompare.heldOutTestCoverage": "{samples} zurückgehaltene Testbeispiele",
    "modelCompare.sharedHeldOutNeedsLineage": "Für diesen ursprünglichen IQB-Export die reduzierte 5-%-CSV und die alte 100-%-Quell-CSV im Überschneidungsbereich hochladen und die Prüfung ausführen. Der ursprüngliche Testsatz mit 5.249 Zeilen enthält Trainingszeilen des reduzierten Modells.",
    "modelCompare.sharedHeldOutReady": "Geprüfter gemeinsamer Testsatz: {samples} Quellzeilen ohne Trainingsbeobachtungen des reduzierten Modells.",
    "modelCompare.mainTrainingPerformanceCoverage": "{samples} Trainingsbeispiele",
    "modelCompare.heldOutBeforeEdits": "Original-Artefakt",
    "modelCompare.heldOutAfterEdits": "Bearbeitetes Artefakt",
    "modelCompare.heldOutDeltaEditedMinusOriginal": "Bearbeitet − Original",
    "modelCompare.heldOutResidualStandardDeviation": "Standardabweichung der Residuen",
    "modelCompare.heldOutMeanError": "Mittlerer Fehler",
    "modelCompare.actualVsPredicted": "Tatsächlich vs. vorhergesagt",
    "modelCompare.actualValues": "Tatsächliche Werte",
    "modelCompare.predictedValues": "Vorhergesagte Werte",
    "modelCompare.actualVsPredictedIdentityLine": "Perfekte Vorhersage",
    "modelCompare.heldOutTestReady":
      "Wähle einen Testsatz und bewerte anschließend alle drei Modellzustände.",
    "modelCompare.heldOutTestUnavailable":
      "Das ausgewählte Artefakt enthält keine zurückgehaltenen Testdaten. Exportiere es nach dem Training mit dieser Version erneut und lade das neue Artefakt hoch.",
    "modelCompare.averagePredictedScore":
      "Vergleich durchschnittlicher Vorhersagen",
    "modelCompare.averagePredictedScoreDescription":
      "Vergleicht das Modell mit vollständigen Daten sowie das unbearbeitete und das bearbeitete Modell mit reduzierten Daten auf der gewählten Referenz; ausgewählte Bearbeitungen und die Sicherheitsgewichtung sind enthalten.",
    "modelCompare.interceptAdjustmentLabel": "Interzept-Anpassung des bearbeiteten Modells",
    "modelCompare.interceptAdjustmentDescription":
      "Addiert diese Konstante zu den bearbeiteten Vorhersagen, auch im zurückgehaltenen Leistungstest. Die oben gezeigte Empfehlung aus den 80-%-Trainingszeilen gleicht die gemessene Änderung aus.",
    "modelCompare.applyInterceptAdjustment": "Anwenden",
    "modelCompare.interceptAdjustmentInvalid": "Gib eine endliche Zahl für die Interzept-Anpassung ein.",
    "modelCompare.originalEditedBaseOffset": "Ursprünglicher Basisoffset des bearbeiteten Modells",
    "modelCompare.activeInterceptAdjustment": "Aktive Anpassung",
    "modelCompare.adjustedEditedBaseOffset": "Angepasster Basisoffset des bearbeiteten Modells",
    "modelCompare.testSetAverage": "Zurückgehaltener Testsatz des Originals",
    "modelCompare.mainDatasetAverage": "80-%-Trainingssatz des Originalmodells",
    "modelCompare.testSetAverageDescription":
      "Beide Artefakte sagen jede zurückgehaltene Feature-Zeile aus dem ursprünglichen Modell-JSON vorher; die Karten zeigen die mittleren Vorhersagen.",
    "modelCompare.mainDatasetAverageDescription":
      "Beide Artefakte sagen die {count} exportierten Trainingszeilen des Originalmodells voraus (sein 80-%-Entwicklungssatz). Die mittlere Änderung wird vor der Interzept-Anpassung berechnet.",
    "modelCompare.mainDatasetAverageUnavailable":
      "Das Originalmodell-Artefakt enthält keine exportierten Trainingszeilen. Exportiere es erneut, um die Änderung auf dem 80-%-Trainingssatz zu berechnen.",
    "modelCompare.developmentMeanEditShift":
      "Mittlere Änderung durch Bearbeitungen vor der Interzept-Anpassung",
    "modelCompare.developmentSuggestedAdjustment":
      "Empfohlene Interzept-Anpassung aus diesen Trainingszeilen",
    "modelCompare.heldOutConfidenceWeighted":
      "Die Vorhersagen auf dem zurückgehaltenen Testsatz verwenden nach Konfidenz gewichtete Expertenbearbeitungen.",
    "modelCompare.heldOutUnweighted":
      "Die Vorhersagen auf dem zurückgehaltenen Testsatz verwenden ungewichtete Expertenbearbeitungen in ihrer eingereichten Stärke.",
    "modelCompare.missingIndicatorsExcludedFromAverage":
      "Erzeugte [missing]-Indikator-Features werden aus diesen Mittelwerten und der Feature-Aufschlüsselung ausgeschlossen.",
    "modelCompare.testSetAverageUnavailable":
      "Das ursprüngliche Modell-JSON enthält keine zurückgehaltenen Testzeilen. Exportiere es nach dem Training erneut, um diese Option zu verwenden.",
    "modelCompare.limitEditedMeanDeviation":
      "Änderung des bearbeiteten Mittelwerts auf 10 % begrenzen",
    "modelCompare.limitEditedMeanDeviationDescription":
      "Skaliert jede ausgewählte Bearbeitung im bearbeiteten Artefakt mit einem gemeinsamen Faktor; gespeicherte Bearbeitungen bleiben unverändert.",
    "modelCompare.editScaleFactor": "×{factor}",
    "modelCompare.meanDeviationScaled":
      "Ausgewählte bearbeitete Offsets werden mit {percent} % ihrer eingereichten Stärke angezeigt, um die Begrenzung einzuhalten.",
    "modelCompare.meanDeviationWithinLimit":
      "Die ausgewählten Bearbeitungen liegen bereits innerhalb der 10-%-Begrenzung.",
    "modelCompare.meanDeviationCannotMeetLimit":
      "Das unbearbeitete rechte Modell weicht bereits um mehr als 10 % ab; allein durch Skalieren der Bearbeitungen kann die Begrenzung nicht erreicht werden.",
    "modelCompare.originalAveragePredictedScore":
      "Original-Artefakt mit vollständigen Daten",
    "modelCompare.editedAveragePredictedScore":
      "Bearbeitetes Artefakt auf gewählter Referenz",
    "modelCompare.averagePredictedScoreDifference":
      "Bearbeitet − Original mit vollständigen Daten",
    "modelCompare.editedMinusReducedAveragePrediction":
      "Unterschied der mittleren Vorhersage: bearbeitetes − unbearbeitetes Modell mit reduzierten Daten",
    "modelCompare.averagePredictionRelativeChange":
      "Relative Veränderung: {change}",
    "modelCompare.trainingMeanDetailsTitle": "Trainingsmittelwerte und Interzeptwerte",
    "modelCompare.trainingMeanDetailsDescription":
      "Aus den Artefakten rekonstruierte mittlere Vorhersagen auf den vorverarbeiteten Trainingszeilen des Modells mit reduzierten Daten ({count} Zeilen), ohne erzeugte [missing]-Indikatoren. Ausgewählte Expertenanpassungen und die Skalierung der Vorschau sind berücksichtigt. Der Wert nach Korrektur verwendet die aktuell angewandte Interzept-Anpassung.",
    "modelCompare.trainingMeanUnavailable":
      "Trainingsmittelwerte sind nicht verfügbar, da dieses Artefakt keine Trainings-Feature-Zeilen enthält. Exportiere das Modell mit reduzierten Daten mit dieser Version erneut und lade es wieder hoch.",
    "modelCompare.reducedTrainingMean": "Mittlere Vorhersage des Modells mit reduzierten Daten",
    "modelCompare.editedTrainingMeanBeforeCorrection":
      "Mittlere Vorhersage nach Expertenanpassung vor Interzept-Korrektur",
    "modelCompare.trainingMeanDifferenceBeforeCorrection":
      "Unterschied durch Expertenanpassung (bearbeitet − reduziert)",
    "modelCompare.editedTrainingMeanAfterCorrection":
      "Mittlere Vorhersage nach Expertenanpassung und angewandter Korrektur",
    "modelCompare.originalFullDataIntercept": "Interzept des Originalmodells mit vollständigen Daten",
    "modelCompare.reducedModelIntercept": "Interzept des Modells mit reduzierten Daten",
    "modelCompare.editedInterceptBeforeCorrection":
      "Interzept nach Expertenanpassung vor Korrektur",
    "modelCompare.appliedInterceptCorrection": "Angewandte Interzept-Korrektur",
    "modelCompare.editedInterceptAfterCorrection":
      "Interzept nach Expertenanpassung und angewandter Korrektur",
    "modelCompare.suggestedMatchingCorrection":
      "Korrektur zum Angleichen des reduzierten Trainingsmittelwerts",
    "modelCompare.suggestedMatchingIntercept":
      "Interzept mit ausgleichender Korrektur",
    "modelCompare.averageContributionsTitle":
      "Änderung der mittleren Vorhersage je Feature",
    "modelCompare.reducedEditedContributionsTitle":
      "Änderung der mittleren Vorhersage je Feature: bearbeitetes − unbearbeitetes Modell mit reduzierten Daten",
    "modelCompare.contributionFeature": "Feature",
    "modelCompare.meanPredictionChangeColumn":
      "Änderung der mittleren Vorhersage (bearbeitet − Original)",
    "modelCompare.reducedEditedMeanPredictionChangeColumn":
      "Änderung der mittleren Vorhersage (bearbeitet − unbearbeitet, reduzierte Daten)",
    "modelCompare.totalMeanDeviationFromOriginal":
      "Gesamtabweichung vom Originalmodell (vollständige Daten)",
    "modelCompare.totalMeanDeviationFromOriginalReduced":
      "Gesamtabweichung vom unbearbeiteten Modell (reduzierte Daten)",
    "modelCompare.featureMeanShiftLimit": "10%-Mittelwertgrenze",
    "modelCompare.meanShiftFeatureLimitDescription":
      "Wähle die zu skalierenden bearbeiteten Features. Für diese Menge wird ein optimaler gemeinsamer Faktor berechnet; alle anderen Feature-Bearbeitungen bleiben unverändert.",
    "modelCompare.chooseMeanShiftFeatures": "Features auswählen",
    "modelCompare.meanShiftFeatureModalTitle":
      "Features für die 10%-Mittelwertgrenze auswählen",
    "modelCompare.meanShiftFeatureModalDescription":
      "Die ausgewählten Features erhalten einen optimalen gemeinsamen Faktor, damit sich ihr ausgewählter Gesamtmittelwert in den 10%-Bereich bewegt.",
    "modelCompare.meanShiftFeatureModalSelection":
      "{selected} von {total} Features ausgewählt",
    "modelCompare.applyMeanShiftFeatureLimit": "Optimalen Faktor anwenden",
    "modelCompare.limitFeatureMeanShift": "Dieses Feature begrenzen",
    "modelCompare.removeFeatureMeanShiftLimit": "Feature-Begrenzung entfernen",
    "modelCompare.selectedContributionAverageView":
      "Die Karten zeigen den Basisoffset plus ausgewählte Feature-Beiträge.",
    "modelCompare.recomputing": "Vorschau wird neu berechnet...",
    "modelCompare.noMatchingFeatures":
      "Keine Feature-Diagramme passen zur aktuellen Suche.",
    "modelCompare.maeVsEditedBase": "Bearbeitet vs bearbeitete Basis",
    "modelCompare.maeVsOriginalBase": "Bearbeitet vs Original-Basis",
    "modelCompare.rmseVsEditedBase": "Bearbeitet vs bearbeitete Basis",
    "modelCompare.rmseVsOriginalBase": "Bearbeitet vs Original-Basis",
    "modelCompare.r2VsEditedBase": "Bearbeitet vs bearbeitete Basis",
    "modelCompare.r2VsOriginalBase": "Bearbeitet vs Original-Basis",
    "modelCompare.maeOriginalVsEdited": "Original vs ausgewählte Bearbeitungen",
    "modelCompare.rmseOriginalVsEdited": "Original vs ausgewählte Bearbeitungen",
    "modelCompare.r2OriginalVsEdited": "Original vs ausgewählte Bearbeitungen",
    "modelCompare.metricImproved": "Verbessert",
    "modelCompare.metricWorsened": "Verschlechtert",
    "modelCompare.metricUnchanged": "Unverändert",
    "modelCompare.reducedVsFullData": "Reduzierte Daten vs vollständige Daten",
    "modelCompare.deltaFromBaseline": "Δ zur reduzierten Daten-Baseline",
    "modelCompare.relativeChange": "Relative Veränderung",
    "modelCompare.leftBase": "Baseline: vollständige Daten",
    "modelCompare.leftEffective": "Original",
    "modelCompare.rightBase": "Baseline: reduzierte Daten",
    "modelCompare.rightEffective": "Bearbeitet",
    "modelCompare.aggregatedParticipantEdits":
      "Zusammengefasste Bearbeitungen der Expert:innen",
    "modelCompare.editLeftSelections": "Original-Auswahl bearbeiten",
    "modelCompare.editRightSelections": "Bearbeitete Auswahl bearbeiten",
  },
};

export const getInitialLanguage = () => {
  if (typeof window === "undefined") return DEFAULT_LANGUAGE;
  const storedLanguage = window.localStorage.getItem(LANGUAGE_STORAGE_KEY);
  return storedLanguage === "de" ? "de" : DEFAULT_LANGUAGE;
};

export const getNumberLocale = (language) =>
  language === "de" ? "de-DE" : "en-US";

export const getDateLocale = (language) =>
  language === "de" ? "de-DE" : "en-US";

const FIXED_DETAIL_QUESTION_VARIABLES = new Set([
  "Sdruckd01a",
  "Sdruckd01b",
  "Sdruckd01c",
  "Sdruckd01d",
  "Sdruckd01e",
  "Ssoe01a",
  "Ssoe01b",
  "Ssoe01c",
  "Ssoe01d",
  "Sind01a",
  "Sind01b",
  "Sind01c",
  "Sind01d",
  "Sword01a",
  "Sword01b",
  "Sword01c",
  "Sword01d",
  "Semod01a",
  "Semod01b",
  "Semod01c",
  "Semod01d",
  "Sumz21a",
  "Sumz21b",
  "Sumz21c",
  "Sumz21d",
  "Sumz21e",
  "Sulkd01a",
  "Sulkd01b",
  "Sulkd01c",
  "Sulkd01d",
  "Sulkd01e",
  "Sulkd01f",
  "Sulkd01g",
]);

const FIXED_PREDICTION_TARGET_KEYS = new Map([
  [
    "Zielvariable – Lesekompetenz Deutsch",
    "prediction.target.germanReading",
  ],
]);

const FIXED_PREDICTION_OPTION_VARIABLES = new Set([
  "TR_TLS1",
  "KBuecher",
  "KDezh16_r",
]);

const getTranslationTemplate = (language, key) =>
  translations[language]?.[key] ?? translations[DEFAULT_LANGUAGE]?.[key];

export const getFeatureDetailQuestionLabel = (language, variable, fallback) =>
  getTranslationTemplate(
    language,
    `shapeFunctions.detailQuestion.${String(variable || "").trim()}`,
  ) || fallback || variable || "";

export const getFeatureDetailAnswerOptionLabel = (
  language,
  sourceVariables,
  value,
  fallback,
) => {
  if (
    !Array.isArray(sourceVariables) ||
    !sourceVariables.some((variable) =>
      FIXED_DETAIL_QUESTION_VARIABLES.has(String(variable || "").trim()),
    )
  ) {
    return fallback || "";
  }

  const numericValue = Number(value);
  if (!Number.isInteger(numericValue)) return fallback || "";

  return (
    getTranslationTemplate(
      language,
      `shapeFunctions.detailAnswerOption.agreement.${numericValue}`,
    ) || fallback || ""
  );
};

export const getPredictionTargetLabel = (language, target, fallback) => {
  const normalizedTarget = String(target || "").trim();
  const key = FIXED_PREDICTION_TARGET_KEYS.get(normalizedTarget);
  return getTranslationTemplate(language, key) || fallback || target || "";
};

const getFeatureProvenanceVariables = (feature) => {
  const provenance = feature?.feature_provenance || {};
  const sourceVariables = Array.isArray(provenance.source_variables)
    ? provenance.source_variables
    : [];
  const sourceDetailVariables = Array.isArray(provenance.source_details)
    ? provenance.source_details.map((detail) => detail?.variable)
    : [];

  return [...sourceVariables, ...sourceDetailVariables]
    .map((variable) => String(variable || "").trim())
    .filter(Boolean);
};

const normalizePredictionOptionValue = (value) => {
  const numericValue = Number(value);
  return Number.isFinite(numericValue)
    ? String(numericValue)
    : String(value || "").trim();
};

export const getPredictionOptionLabel = (language, feature, value, fallback) => {
  const sourceVariables = getFeatureProvenanceVariables(feature);
  const normalizedValue = normalizePredictionOptionValue(value);

  if (
    sourceVariables.some((variable) =>
      FIXED_DETAIL_QUESTION_VARIABLES.has(variable),
    )
  ) {
    return (
      getTranslationTemplate(
        language,
        `shapeFunctions.detailAnswerOption.agreement.${normalizedValue}`,
      ) || fallback || ""
    );
  }

  const sourceVariable = sourceVariables.find((variable) =>
    FIXED_PREDICTION_OPTION_VARIABLES.has(variable),
  );
  return (
    getTranslationTemplate(
      language,
      `prediction.option.${sourceVariable}.${normalizedValue}`,
    ) || fallback || ""
  );
};

export const createTranslator = (language) => (key, variables = {}) => {
  const template = getTranslationTemplate(language, key) ?? key;

  return template.replace(/\{(\w+)\}/g, (_, variableName) => {
    const value = variables[variableName];
    return value === undefined || value === null ? "" : String(value);
  });
};
