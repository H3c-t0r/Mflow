package org.mlflow.tracking;

import org.mlflow.api.proto.Service.*;
import org.mlflow.tracking.utils.DatabricksContext;
import org.mlflow.tracking.utils.MlflowTagConstants;

import java.util.*;
import java.util.function.Consumer;

/**
 * Main entrypoint used to start MLflow runs to log to. This is a higher level interface than
 * {@code MlflowClient} and provides convenience methods to keep track of active runs and to set
 * default tags on runs which are created through {@code MlflowContext}
 *
 * On construction, MlflowContext will choose a default experiment ID to log to depending on your
 * environment. To log to a different experiment, use {@link #setExperimentId(String)}} or
 * {@link #setExperimentName(String)}
 */
public class MlflowContext {
  private MlflowClient client;
  private String experimentId;

  /**
   * Constructs a {@code MlflowContext} with a MlflowClient based on the MLFLOW_TRACKING_URI
   * environment variable.
   */
  public MlflowContext() {
    this(new MlflowClient());
  }

  /**
   * Constructs a {@code MlflowContext} which points to the specified trackingUri.
   *
   * @param trackingUri The URI to log to.
   */
  public MlflowContext(String trackingUri) {
    this(new MlflowClient(trackingUri));
  }

  /**
   * Constructs a {@code MlflowContext} which points to the specified trackingUri.
   *
   * @param client The client used to log runs.
   */
  public MlflowContext(MlflowClient client) {
    this.client = client;
    this.experimentId = getDefaultExperimentId();
  }

  /**
   * Returns the client used to log runs.
   *
   * @return the client used to log runs.
   */
  public MlflowClient getClient() {
    return this.client;
  }

  /**
   * Sets the experiment to log runs to by name.
   * @param experimentName the name of the experiment to log runs to.
   * @throws IllegalArgumentException if the experiment name does not match an existing experiment
   */
  public void setExperimentName(String experimentName) throws IllegalArgumentException {
    Optional<Experiment> experimentOpt = client.getExperimentByName(experimentName);
    if (!experimentOpt.isPresent()) {
      throw new IllegalArgumentException(
        String.format("%s is not a valid experiment", experimentName));
    }
    experimentId = experimentOpt.get().getExperimentId();
  }

  /**
   * Sets the experiment to log runs to by ID.
   * @param experimentId the id of the experiment to log runs to.
   */
  public void setExperimentId(String experimentId) {
    this.experimentId = experimentId;
  }

  /**
   * Returns the experiment ID we are logging to.
   *
   * @return the experiment ID we are logging to.
   */
  public String getExperimentId() {
    return this.experimentId;
  }

  /**
   * Starts a MLflow run. To log data to newly created MLflow run see the methods on
   * {@link ActiveRun}. MLflow runs should be ended using {@link ActiveRun#endRun()}
   *
   * @param runName The name of this run.
   * @return An {@code ActiveRun} object to log data to.
   */
  public ActiveRun startRun(String runName) {
    return startRun(runName, null);
  }

  /**
   * Like {@link #startRun(String)} but sets the {@code mlflow.parentRunId} tag in order to create
   * nested runs.
   *
   * @param runName The name of this run.
   * @param parentRunId The ID of this run's parent
   * @return An {@code ActiveRun} object to log data to.
   */
  public ActiveRun startRun(String runName, String parentRunId) {
    Map<String, String> tags = new HashMap<>();
    tags.put(MlflowTagConstants.RUN_NAME, runName);
    tags.put(MlflowTagConstants.USER, System.getProperty("user.name"));
    tags.put(MlflowTagConstants.SOURCE_TYPE, "LOCAL");
    if (parentRunId != null) {
      tags.put(MlflowTagConstants.PARENT_RUN_ID, parentRunId);
    }

    // Add tags from DatabricksContext if they exist
    DatabricksContext databricksContext = DatabricksContext.createIfAvailable();
    if (databricksContext != null) {
      tags.putAll(databricksContext.getTags());
    }

    CreateRun.Builder createRunBuilder = CreateRun.newBuilder()
      .setExperimentId(experimentId)
      .setStartTime(System.currentTimeMillis());
    for (Map.Entry<String, String> tag: tags.entrySet()) {
      createRunBuilder.addTags(
        RunTag.newBuilder().setKey(tag.getKey()).setValue(tag.getValue()).build());
    }
    RunInfo runInfo = client.createRun(createRunBuilder.build());

    ActiveRun newRun = new ActiveRun(runInfo, client);
    return newRun;
  }

  /**
   * Like {@link #startRun(String)} but will terminate the run after the activeRunFunction is
   * executed.
   *
   * @param runName The name of this run.
   * @param activeRunFunction A function which takes an {@code ActiveRun} and logs data to it.
   */
  public void withActiveRun(String runName, Consumer<ActiveRun> activeRunFunction) {
    ActiveRun newRun = startRun(runName);
    try {
      activeRunFunction.accept(newRun);
    } catch(Exception e) {
      newRun.endRun(RunStatus.FAILED);
      return;
    }
    newRun.endRun(RunStatus.FINISHED);
  }

  private static String getDefaultExperimentId() {
    DatabricksContext databricksContext = DatabricksContext.createIfAvailable();
    if (databricksContext != null && databricksContext.isInDatabricksNotebook()) {
      String notebookId = databricksContext.getNotebookId();
      if (notebookId != null) {
        return notebookId;
      }
    }
    return MlflowClient.DEFAULT_EXPERIMENT_ID;
  }
}
