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
      "Configure how categorical x-axis values are displayed for all users.",
    "shapeFunctions.treatAsCategorical":
      "Treat this chart as categorical",
    "shapeFunctions.treatAsCategoricalHint":
      "Use bars with discrete category labels instead of a continuous line.",
    "shapeFunctions.cannotConvertCategorical":
      "This feature has too many or non-discrete values and cannot be converted to a categorical chart.",
    "shapeFunctions.xAxisValueLabels": "X-Axis Value Labels",
    "shapeFunctions.noCategoricalValues":
      "No categorical values available for mapping.",
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
    "shapeFunctions.chartMappingButton": "Chart Mapping",
    "shapeFunctions.enlargeChart": "Enlarge chart",
    "shapeFunctions.submitFeature": "Submit {feature}",
    "shapeFunctions.dataDistribution": "Data Distribution",
    "shapeFunctions.showDistribution": "Show distribution",
    "shapeFunctions.hideDistribution": "Hide distribution",
    "shapeFunctions.countAxisLabel": "Count",
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
    "shapeFunctions.detailsSourceQuestions": "Source questions",
    "shapeFunctions.detailsAnswerScale": "Answer scale",
    "shapeFunctions.detailsTransformation": "Transformation",
    "shapeFunctions.detailsMissingHandling": "Missing values",
    "shapeFunctions.detailsRationale": "Why this feature was selected",
    "shapeFunctions.detailsSourceCount": "{count} source questions",
    "shapeFunctions.detailsRawSource": "Raw source value",
    "shapeFunctions.detailsItemMean": "Item mean",
    "shapeFunctions.detailsIqbScale": "IQB scale",
    "shapeFunctions.detailsUnknownConstruction": "Derived feature",

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
    "combined.hideUserOverlay": "Hide User Overlay",
    "combined.legend": "Legend:",
    "combined.original": "Original",
    "combined.combined": "Combined",
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
    "superadmin.createUser": "Create User",
    "superadmin.professionPlaceholder": "Enter a profession",
    "superadmin.inviteLink": "Invite Link",
    "superadmin.generateInvite": "Generate Invite",
    "superadmin.inviteCopied": "Invite link copied to clipboard.",
    "superadmin.expires": "Expires",
    "superadmin.inviteDescription":
      "Generate an invite link to allow new registrations.",
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
      "Choose whether the exported model artifact should also include saved shape-function edits tied to the current model.",
    "superadmin.exportIncludeEdits":
      "Include saved shape-function edits",
    "superadmin.exportConfirmAction": "Export Artifact",
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
    "common.reset": "Zuruecksetzen",
    "common.close": "Schliessen",
    "common.delete": "Loeschen",
    "common.target": "Ziel",
    "common.unknown": "Unbekannt",

    "header.subtitle": "Interaktiver GAM-Editor",
    "header.modelTrained": "Modell trainiert",
    "header.modelNotTrained": "Modell nicht trainiert",
    "header.recordsLoaded": "{count} Datensaetze geladen",

    "login.title": "Der interaktive GAM-Editor",
    "login.subtitleLogin": "Mit deinen Zugangsdaten anmelden",
    "login.subtitleRegister": "Mit einem Einladungslink registrieren",
    "login.usernamePlaceholder": "Dein Benutzername",
    "login.passwordPlaceholder": "Dein Passwort",
    "login.professionPlaceholder": "Dein Beruf",
    "login.registerInviteOnly":
      "Eine Registrierung ist nur ueber einen Einladungslink moeglich.",
    "login.error.enterCredentials":
      "Bitte Benutzername und Passwort eingeben",
    "login.error.professionRequired": "Bitte gib deinen Beruf ein",
    "login.error.inviteRequired":
      "Die Registrierung erfordert einen gueltigen Einladungslink",
    "login.error.loginFailed": "Anmeldung fehlgeschlagen",
    "login.error.registerFailed": "Registrierung fehlgeschlagen",
    "login.error.serverUnavailable":
      "Der Server ist nicht erreichbar. Bitte versuche es erneut.",
    "login.submitSigningIn": "Anmeldung laeuft...",
    "login.submitRegistering": "Registrierung laeuft...",
    "login.submitSignIn": "Anmelden",
    "login.submitRegister": "Registrieren",

    "app.loggedInAs": "Angemeldet als",
    "app.viewCombinedResults": "Kombinierte Ergebnisse anzeigen",
    "app.superadmin": "Superadmin",
    "app.logout": "Abmelden",
    "app.footerDescription":
      "GAMWeaver - Interaktiver GAM-Editor fuer interaktive Modellierung und Erklaerbarkeit. Entwickelt von {name} im Rahmen einer Masterarbeit.",
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
      "Shape-Funktionen konnten nicht zurueckgesetzt werden",
    "app.error.resetFeature":
      "Feature konnte nicht zurueckgesetzt werden",
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
      "Das Hochladen/Laden von Datensaetzen und die Trainingssteuerung sind auf den Superadmin beschraenkt.",
    "training.dataLoaded": "Daten geladen",
    "training.modelTrainedLabel": "Modell trainiert",
    "training.activeDataset": "Aktiver Datensatz",
    "training.stepLoadData": "Daten laden",
    "training.stepLoaded": "Geladen",
    "training.loadDescription":
      "CSV-Datensatz hochladen, Vorhersageziel auswaehlen und zu importierende Spalten festlegen.",
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
    "training.uploadInspect": "Hochladen und Spalten pruefen",
    "training.columnsDetected": "Erkannte Spalten",
    "training.targetColumn": "Zielspalte",
    "training.featureColumns": "Feature-Spalten",
    "training.expandFeatureColumns": "Feature-Spalten erweitern",
    "training.collapseFeatureColumns": "Feature-Spalten verkleinern",
    "training.selectAll": "Alle waehlen",
    "training.clearAll": "Auswahl loeschen",
    "training.target": "Ziel",
    "training.selectedFeatures": "Ausgewaehlte Features",
    "training.loadDataset": "Datensatz laden",
    "training.error.chooseCsvFirst": "Bitte zuerst eine CSV-Datei auswaehlen.",
    "training.error.uploadFailed": "Datensatz konnte nicht hochgeladen werden",
    "training.error.loadFailed": "Datensatz konnte nicht geladen werden",
    "training.error.uploadNeedsSelections":
      "Bitte eine CSV hochladen, eine Zielspalte waehlen und mindestens eine Feature-Spalte auswaehlen.",
    "training.progress.loadingData": "Daten werden geladen...",
    "training.progress.training":
      "Modell wird trainiert... Das kann einen Moment dauern.",
    "training.uploading": "Wird hochgeladen...",
    "training.loadingDataset": "Wird geladen...",
    "training.training": "Training laeuft...",
    "training.statusDataLoaded": "Daten geladen",
    "training.statusModelTrained": "Modell trainiert",
    "training.statusModelSource": "Modellquelle",
    "training.modelSourceImported": "importiert",
    "training.modelSourceTrained": "trainiert",
    "training.importedAnalyticsUnavailable":
      "Analysen fuer importierte Modelle bleiben nicht verfuegbar, bis der Superadmin einen kompatiblen Datensatz laedt.",

    "dataSummary.title": "Datenuebersicht",
    "dataSummary.empty": "Daten laden, um Zusammenfassungsstatistiken zu sehen",
    "dataSummary.totalRecords": "Gesamtanzahl Datensaetze",
    "dataSummary.targetVariable": "Zielvariable",
    "dataSummary.targetFallback": "Ziel",
    "dataSummary.mean": "Mittelwert",
    "dataSummary.stdDev": "Std.-Abw.",
    "dataSummary.min": "Min",
    "dataSummary.max": "Max",
    "dataSummary.features": "Features",

    "prediction.title": "Vorhersage erstellen",
    "prediction.description":
      "Feature-Werte des Datensatzes setzen und eine Vorhersage ausfuehren. Die Eingaben werden aus dem aktiven Datensatzschema generiert.",
    "prediction.trainFirst":
      "Bitte zuerst das Modell trainieren, bevor Vorhersagen erstellt werden.",
    "prediction.noSchema":
      "Kein Feature-Schema verfuegbar. Bitte den Superadmin, einen Datensatz zu laden.",
    "prediction.predicting": "Vorhersage laeuft...",
    "prediction.predictTarget": "{target} vorhersagen",
    "prediction.predictedTarget": "Vorhergesagtes {target}",

    "shapeFunctions.title": "Shape-Funktionen",
    "shapeFunctions.emptyDescription":
      "Trainiere das Modell, um Feature-Shape-Funktionen zu sehen.",
    "shapeFunctions.sharedDescription":
      "Shape-Funktionen zeigen, wie jedes Feature die Vorhersage beeinflusst.",
    "shapeFunctions.gridDescription":
      "Diese Diagramme zeigen, wie jedes Feature die Vorhersage der Fahrradvermietung beeinflusst. Werte ueber null erhoehen die Vorhersage, Werte darunter senken sie.",
    "shapeFunctions.effectOnPrediction":
      "Effekt auf die Vorhersage",
    "shapeFunctions.interactiveTitle": "Interaktive Shape-Funktionen",
    "shapeFunctions.interactiveEmptyDescription":
      "Trainiere das Modell, um Feature-Shape-Funktionen zu sehen und zu bearbeiten.",
    "shapeFunctions.interactiveEditDescription":
      "Bearbeite Punkte und klicke dann bei jedem Feature auf Senden, um mit deiner Sicherheit zu speichern.",
    "shapeFunctions.interactiveViewDescription":
      "Aktiviere den Bearbeitungsmodus, um Shape-Funktionen interaktiv zu aendern.",
    "shapeFunctions.resetAll": "Alles zuruecksetzen",
    "shapeFunctions.syncAxes": "Achsen synchronisieren",
    "shapeFunctions.axesSynced": "Achsen synchronisiert",
    "shapeFunctions.syncAxesHintOn":
      "Achsen sind synchronisiert - klicken fuer individuelle Skalierung",
    "shapeFunctions.syncAxesHintOff":
      "Klicken, um alle Diagrammachsen auf dieselbe Skala zu synchronisieren",
    "shapeFunctions.enableEditing": "Bearbeitung aktivieren",
    "shapeFunctions.editingModeOn": "Bearbeitungsmodus AN",
    "shapeFunctions.editHelp":
      "Bei numerischen Features klicken und ziehen, um Bearbeitungen entlang der Kurve zu pinseln. Mit dem Weich-Hart-Regler steuerst du, wie viel der Kurve beeinflusst wird. Bei kategorialen Features einen Punkt anfahren und nach oben oder unten ziehen. Doppelklick fuer eine praezise Eingabe. Wenn du mit einem Feature fertig bist, klicke auf Senden und bewerte deine Sicherheit.",
    "shapeFunctions.lineBrushHardness": "Pinselhaerte der Linie",
    "shapeFunctions.soft": "Weich",
    "shapeFunctions.hard": "Hart",
    "shapeFunctions.original": "Original",
    "shapeFunctions.edited": "Bearbeitet",
    "shapeFunctions.current": "Aktuell",
    "shapeFunctions.comparison": "Vergleich",
    "shapeFunctions.editPoints": "Punkte bearbeiten",
    "shapeFunctions.effect": "Effekt",
    "shapeFunctions.draggingSuffix": "(wird gezogen...)",
    "shapeFunctions.editedSuffix": "(bearbeitet)",
    "shapeFunctions.setEffectValue": "Effektwert festlegen",
    "shapeFunctions.brushingAtX": "Pinsel bei x = {value}",
    "shapeFunctions.releaseToApply":
      "Loslassen, um geglaetteten Strich anzuwenden: {value}",
    "shapeFunctions.draggingPoint": "Punkt {index} wird gezogen",
    "shapeFunctions.releaseToSetValue":
      "Loslassen, um Wert zu setzen: {value}",
    "shapeFunctions.clickDragHint":
      "Klicken und ziehen, um die Linie zu bearbeiten. Doppelklick fuer eine praezise Eingabe.",
    "shapeFunctions.pointSelected": "Punkt {index} ausgewaehlt",
    "shapeFunctions.preciseValueHint":
      "Doppelklick fuer praezisen Wert",
    "shapeFunctions.hoverSelectHint":
      "Mit der Maus ueber einen Punkt fahren, um ihn auszuwaehlen",
    "shapeFunctions.submitEditTitle":
      "Bearbeitung fuer {feature} senden",
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
      "Bitte gib eine Beschreibung fuer deine Bearbeitung ein.",
    "shapeFunctions.submitEdit": "Bearbeitung senden",
    "shapeFunctions.chartSettingsSaveError":
      "Diagrammeinstellungen konnten nicht gespeichert werden",
    "shapeFunctions.chartMappingTitle":
      "Diagrammzuordnung: {feature}",
    "shapeFunctions.chartMappingDescription":
      "Konfiguriere, wie kategoriale Werte auf der x-Achse fuer alle Benutzer angezeigt werden.",
    "shapeFunctions.treatAsCategorical":
      "Dieses Diagramm als kategorial behandeln",
    "shapeFunctions.treatAsCategoricalHint":
      "Balken mit diskreten Kategorien statt einer kontinuierlichen Linie verwenden.",
    "shapeFunctions.cannotConvertCategorical":
      "Dieses Feature hat zu viele oder nicht-diskrete Werte und kann nicht in ein kategoriales Diagramm umgewandelt werden.",
    "shapeFunctions.xAxisValueLabels":
      "Wertebeschriftungen der X-Achse",
    "shapeFunctions.noCategoricalValues":
      "Keine kategorialen Werte fuer die Zuordnung verfuegbar.",
    "shapeFunctions.displayLabelFor":
      "Anzeigelabel fuer {value}",
    "shapeFunctions.mappingAvailableAfterCategorical":
      "Label-Zuordnungen sind verfuegbar, sobald dieses Feature als kategoriales Diagramm angezeigt wird.",
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
    "shapeFunctions.chartMappingButton": "Diagrammzuordnung",
    "shapeFunctions.enlargeChart": "Diagramm vergroessern",
    "shapeFunctions.submitFeature": "{feature} senden",
    "shapeFunctions.dataDistribution": "Datenverteilung",
    "shapeFunctions.showDistribution": "Verteilung anzeigen",
    "shapeFunctions.hideDistribution": "Verteilung ausblenden",
    "shapeFunctions.countAxisLabel": "Anzahl",
    "shapeFunctions.distributionEmpty": "Keine Verteilungsdaten verfuegbar.",
    "shapeFunctions.openFeatureDetails": "Feature-Details oeffnen",
    "shapeFunctions.detailsButton": "Details",
    "shapeFunctions.detailsDrawerTitle": "Feature-Details",
    "shapeFunctions.detailsSummary":
      "Zeigt, wie dieses Feature im Datensatz konstruiert wurde.",
    "shapeFunctions.detailsUnavailable":
      "Fuer dieses Feature sind keine Metadaten zur Herkunft verfuegbar.",
    "shapeFunctions.detailsCategory": "Kategorie",
    "shapeFunctions.detailsConstruction": "Konstruktion",
    "shapeFunctions.detailsSourceVariables": "Quellvariablen",
    "shapeFunctions.detailsSourceQuestions": "Quellfragen",
    "shapeFunctions.detailsAnswerScale": "Antwortskala",
    "shapeFunctions.detailsTransformation": "Transformation",
    "shapeFunctions.detailsMissingHandling": "Fehlende Werte",
    "shapeFunctions.detailsRationale":
      "Warum dieses Feature ausgewaehlt wurde",
    "shapeFunctions.detailsSourceCount": "{count} Quellfragen",
    "shapeFunctions.detailsRawSource": "Direkter Quellwert",
    "shapeFunctions.detailsItemMean": "Itemmittelwert",
    "shapeFunctions.detailsIqbScale": "IQB-Skalenwert",
    "shapeFunctions.detailsUnknownConstruction": "Abgeleitetes Feature",

    "predictionComparison.title": "Vorhersagevergleich",
    "predictionComparison.empty":
      "Bearbeite Shape-Funktionen und wende Aenderungen an, um den Vorhersagevergleich zu sehen.",
    "predictionComparison.subtitleWithEdits":
      "Vergleiche Vorhersagen zwischen dem urspruenglichen IGANN-Modell und deiner bearbeiteten Version.",
    "predictionComparison.subtitleWithoutEdits":
      "Bearbeite Shape-Funktionen, um zu sehen, wie sich Vorhersagen aendern.",
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
      "Melde dich an, um Farben sitzungsuebergreifend zu speichern.",
    "predictionComparison.originalTrace": "IGANN (Original)",
    "predictionComparison.editedTrace":
      "IGANN Interaktiv (Bearbeitet)",
    "predictionComparison.perfectPrediction":
      "Perfekte Vorhersage",
    "predictionComparison.plotTitle": "Vorhersagen vs. Ist-Werte",
    "predictionComparison.plotTitleWithEdits":
      "Vorhersagen vs. Ist-Werte (Original vs. Bearbeitet)",
    "predictionComparison.actualBikeRentals":
      "Tatsaechliche Fahrradvermietungen",
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
      "Diese Seite zeigt den aggregierten Effekt aller Benutzerbearbeitungen auf das GAM-Modell. Der Offset jedes Punkts wird ueber alle Benutzer gemittelt, die ihn bearbeitet haben.",
    "combined.participatingUsers":
      "Teilnehmende Benutzer ({count})",
    "combined.noUsers": "Noch keine Benutzer.",
    "combined.originalModel": "Originalmodell",
    "combined.combinedUserEdits":
      "Kombinierte Benutzerbearbeitungen",
    "combined.rmse": "RMSE",
    "combined.mae": "MAE",
    "combined.scatterTitle":
      "Vorhergesagt vs. Original (Streudiagramm)",
    "combined.scatterDescription":
      "Punkte naeher an der Diagonalen zeigen bessere Vorhersagen. Vergleiche, wie sich kombinierte Benutzerbearbeitungen auf die Genauigkeit auswirken.",
    "combined.formTitle": "Mit kombinierten Bearbeitungen vorhersagen",
    "combined.formDescription":
      "Erstelle eine Vorhersage mit dem Modell, auf das alle kombinierten Benutzerbearbeitungen angewendet wurden. Das Ergebnis spiegelt die aggregierten Aenderungen der Shape-Funktionen aller Benutzer wider.",
    "combined.predictButton":
      "{target} vorhersagen (kombiniertes Modell)",
    "combined.predictedResult":
      "Vorhergesagtes {target} (kombinierte Bearbeitungen)",
    "combined.predictionFailed": "Vorhersage fehlgeschlagen",
    "combined.shapeFunctionsNone":
      "Keine Shape-Funktionsdaten verfuegbar.",
    "combined.shapeFunctionsTitle":
      "Shape-Funktionen: Original vs. kombinierte Bearbeitungen",
    "combined.weightingToggleTitle":
      "Umschalten, ob die kombinierte Linie eine sicherheitsgewichtete Mittelung oder einen einfachen Mittelwert verwendet",
    "combined.weightingOn": "Gewichtung: An",
    "combined.weightingOff": "Gewichtung: Aus",
    "combined.showUserOverlay": "Benutzer-Overlay anzeigen",
    "combined.hideUserOverlay": "Benutzer-Overlay ausblenden",
    "combined.legend": "Legende:",
    "combined.original": "Original",
    "combined.combined": "Kombiniert",
    "combined.noEdits":
      "Es wurden noch keine Benutzerbearbeitungen vorgenommen. Die Diagramme unten zeigen die urspruenglichen Shape-Funktionen.",
    "combined.modified": "Geaendert",
    "combined.effect": "Effekt",
    "combined.editLogsUnavailable":
      "Bearbeitungsprotokolle sind voruebergehend nicht verfuegbar: {error}",
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
    "combined.xSummary": "X-Uebersicht",
    "combined.deleteSubmission": "Einreichung loeschen",
    "combined.xValue": "X-Wert",
    "combined.rawInput": "Roheingabe",
    "combined.weighted": "Gewichtet",
    "combined.originalLinePreview": "Originale Linie",
    "combined.weightedLinePreview": "Gewichtete bearbeitete Linie",
    "combined.noSubmissions":
      "Keine Einreichungen fuer dieses Feature verfuegbar.",
    "combined.deleteDisabled":
      "Destruktive Aktionen sind in dieser Demo deaktiviert.",
    "combined.deleteReasonRequired":
      "Bitte gib einen Grund fuer das Loeschen dieser Bearbeitung an.",
    "combined.noDeletionTarget":
      "Kein Loeschziel fuer diese Einreichung verfuegbar",
    "combined.deleteFailed":
      "Einreichung konnte nicht geloescht werden",
    "combined.loadEditLogsFailed":
      "Bearbeitungsprotokolle konnten nicht geladen werden",
    "combined.loadDataFailed": "Daten konnten nicht geladen werden",
    "combined.deleteModalTitle": "Einreichung loeschen",
    "combined.deleteModalDescription":
      "Du bist dabei, eine eingereichte Kurvenbearbeitung von {user} zu loeschen.",
    "combined.feature": "Feature",
    "combined.whyRemoved":
      "Warum wird diese Einreichung entfernt?",
    "combined.deleteReasonPlaceholder":
      "Gib einen Grund fuer das Entfernen dieser Einreichung an...",
    "combined.deleting": "Loeschen...",

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
      "Datenbank konnte nicht zurueckgesetzt werden",
    "superadmin.error.exportModel":
      "Modell konnte nicht exportiert werden",
    "superadmin.error.importModel":
      "Modell konnte nicht importiert werden",
    "superadmin.compareUploadError":
      "Vergleichsdatensatz konnte nicht hochgeladen werden",
    "superadmin.compareLoadError":
      "Vergleichsdatensatz konnte nicht geladen werden",
    "superadmin.title": "Superadmin-Uebersicht",
    "superadmin.openCombined": "Kombinierte Ergebnisse oeffnen",
    "superadmin.compareDatasets": "Datensaetze vergleichen",
    "superadmin.compareDatasetsDescription":
      "Lade einen zweiten Datensatz hoch, trainiere ihn mit der Estimator-Anzahl des primaeren Modells und lege seine Shape-Funktionen ueber die interaktiven Diagramme.",
    "superadmin.compareModalDescription":
      "Der Vergleichsdatensatz muss dieselbe Zielspalte und dieselben ausgewaehlten Feature-Spalten wie der primaere Datensatz verwenden.",
    "superadmin.uploadAndSelectComparison":
      "Hochladen und Features waehlen",
    "superadmin.trainComparison": "Vergleichsmodell trainieren",
    "superadmin.primaryEstimators": "Primaere Estimatoren",
    "superadmin.comparisonDataset": "Vergleichsdatensatz",
    "superadmin.noComparisonDataset": "Kein Vergleichsdatensatz geladen",
    "superadmin.comparisonTrained": "Vergleich trainiert",
    "superadmin.loadComparisonDataset": "Vergleichsdatensatz laden",
    "superadmin.createUser": "Benutzer erstellen",
    "superadmin.professionPlaceholder": "Beruf eingeben",
    "superadmin.inviteLink": "Einladungslink",
    "superadmin.generateInvite": "Einladung erzeugen",
    "superadmin.inviteCopied":
      "Einladungslink wurde in die Zwischenablage kopiert.",
    "superadmin.expires": "Laeuft ab",
    "superadmin.inviteDescription":
      "Einen Einladungslink erzeugen, um neue Registrierungen zu ermoeglichen.",
    "superadmin.systemReset": "System zuruecksetzen",
    "superadmin.systemResetDescription":
      "Entfernt dauerhaft alle Benutzer, alle gespeicherten Bearbeitungen, hochgeladene CSV-Dateien und den extrahierten Feature-Zustand der aktuellen Backend-Umgebung.",
    "superadmin.resetDatabase": "Datenbank zuruecksetzen",
    "superadmin.creating": "Wird erstellt...",
    "superadmin.users": "Benutzer",
    "superadmin.loadingUsers": "Benutzer werden geladen...",
    "superadmin.noUsers": "Noch keine Benutzer.",
    "superadmin.created": "Erstellt",
    "superadmin.badge": "Superadmin",
    "superadmin.resetConfirmTitle": "Datenbank zuruecksetzen?",
    "superadmin.resetConfirmBody":
      "Dadurch werden Benutzer, Bearbeitungen, hochgeladene CSV-Dateien und extrahierter Feature-Zustand dauerhaft geloescht. Diese Aktion kann nicht rueckgaengig gemacht werden.",
    "superadmin.resetting": "Zuruecksetzen laeuft...",
    "superadmin.resetEverything": "Ja, alles zuruecksetzen",
    "superadmin.modelTransferTitle": "Modelltransfer",
    "superadmin.modelTransferDescription":
      "Exportiere das aktive trainierte Modell als JSON oder importiere ein zuvor exportiertes Modellartefakt.",
    "superadmin.exporting": "Export laeuft...",
    "superadmin.exportModel": "Modell exportieren",
    "superadmin.importing": "Import laeuft...",
    "superadmin.importModel": "Modell importieren",
    "superadmin.modelTransferWarning":
      "Beim Importieren eines Modells wird das aktuelle Basismodell ersetzt und gespeicherte Benutzerbearbeitungen, die mit dem vorherigen Modell verknuepft sind, werden geloescht.",
    "superadmin.exportSuccess": "Modell wurde als {filename} exportiert.",
    "superadmin.exportSuccessWithEdits":
      "Modell und gespeicherte Shape-Function-Bearbeitungen wurden als {filename} exportiert.",
    "superadmin.importSuccess":
      "Modell wurde erfolgreich importiert.",
    "superadmin.importSuccessWithEdits":
      "Modell und gespeicherte Shape-Function-Bearbeitungen wurden erfolgreich fuer {users} Benutzer mit {submissions} Einreichungen importiert.",
    "superadmin.exportConfirmTitle": "Modellartefakt exportieren?",
    "superadmin.exportConfirmBody":
      "Waehle, ob das exportierte Modellartefakt auch gespeicherte Shape-Function-Bearbeitungen des aktuellen Modells enthalten soll.",
    "superadmin.exportIncludeEdits":
      "Gespeicherte Shape-Function-Bearbeitungen einschliessen",
    "superadmin.exportConfirmAction": "Artefakt exportieren",
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

export const createTranslator = (language) => (key, variables = {}) => {
  const template =
    translations[language]?.[key] ?? translations[DEFAULT_LANGUAGE]?.[key] ?? key;

  return template.replace(/\{(\w+)\}/g, (_, variableName) => {
    const value = variables[variableName];
    return value === undefined || value === null ? "" : String(value);
  });
};
