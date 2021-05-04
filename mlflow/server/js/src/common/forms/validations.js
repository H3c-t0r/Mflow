import { MlflowService } from '../../experiment-tracking/sdk/MlflowService';
import { Services as ModelRegistryService } from '../../model-registry/services';
import { wrapDeferred } from '../utils/ActionUtils';

export const getExperimentNameValidator = (getExistingExperimentNames) => {
  return (rule, value, callback) => {
    if (value.length === 0) {
      // no need to execute below validations when no value is entered
      // eslint-disable-next-line callback-return
      callback(undefined);
    } else if (getExistingExperimentNames().includes(value)) {
      // getExistingExperimentNames returns the names of all active experiments
      // check whether the passed value is part of the list
      // eslint-disable-next-line callback-return
      callback(`Experiment "${value}" already exists.`);
    } else {
      // on-demand validation whether experiment already exists in deleted state
      wrapDeferred(MlflowService.getExperimentByName, { experiment_name: value })
        .then((res) =>
          callback(`Experiment "${value}" already exists in deleted state.
                                 You can restore the experiment, or permanently delete the
                                 experiment from the .trash folder (under tracking server's
                                 root folder) in order to use this experiment name again.`),
        )
        .catch((e) => callback(undefined)); // no experiment returned
    }
  };
};

export const getExperimentIdValidator = (getCurrentExperimentId, getExistingExperimentIds) => {
  return (rule, value, callback) => {
    if (value.length === 0) {
      // no need to execute below validations when no value is entered
      // eslint-disable-next-line callback-return
      callback(undefined);
    } else if (isNaN(value)) {
      callback(`Experiment ID must be an integer`);
    } else if (getCurrentExperimentId() == value) {
      callback(`"${value}" is the current experiment ID`);
    } else if (!getExistingExperimentIds().includes(value)) {
      // getExistingExperimentNames returns the names of all active experiments
      // check whether the passed value is part of the list
      // eslint-disable-next-line callback-return
      callback(`Experiment ID "${value}" does not exists.`);
    } else {
      callback(undefined); // experiment id exists
    }
  };
};

export const modelNameValidator = (rule, name, callback) => {
  if (name.length === 0) {
    callback(undefined);
    return;
  }
  ModelRegistryService.getRegisteredModel({ data: { name } })
    .then(() => callback(`Model "${name}" already exists.`))
    .catch((e) => callback(undefined));
};
