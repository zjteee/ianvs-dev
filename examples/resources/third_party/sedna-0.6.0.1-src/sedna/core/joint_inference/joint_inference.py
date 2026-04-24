# Copyright 2021 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# `os` 用于检查本地模型路径是否存在。
import os
# `deepcopy` 用于保护原始推理结果，避免后续 callback / reporter 误改原对象。
from copy import deepcopy

# 获取宿主机可达 IP，用于本地服务绑定或默认云端客户端回连。
from sedna.common.utils import get_host_ip
# ClassFactory 用于按名称加载回调、HEM 等注册类。
from sedna.common.class_factory import ClassFactory, ClassType
# InferenceServer 负责把 BigModelService 暴露成 RESTful 推理服务。
from sedna.service.server import InferenceServer
# ModelClient 用于访问远端大模型；LCReporter 用于向控制面上报统计信息。
from sedna.service.client import ModelClient, LCReporter
# K8sResourceKind 用于标识当前 Job 的资源类型。
from sedna.common.constant import K8sResourceKind
# JobBase 是 Sedna 服务 / 任务类的统一基类，封装了配置与路径等通用逻辑。
from sedna.core.base import JobBase
# `re` 用于判断 model_path 是否是 HuggingFace 仓库名而不是本地路径。
import re

# 允许的 HuggingFace 仓库名形式，例如 `Qwen/Qwen2.5-7B-Instruct`。
HUGGINGFACE_PATH_PATTERN = r'^[a-zA-Z0-9][\w\-]*/[a-zA-Z0-9][\w\-\.]*$'

# 当前文件对外暴露的主要类。
__all__ = ("JointInference", "BigModelService")


class BigModelService(JobBase):
    """
    Large model services implemented
    Provides RESTful interfaces for large-model inference.

    Parameters
    ----------
    estimator : Instance, big model
        An instance with the high-level API that greatly simplifies
        machine learning programming. Estimators encapsulate training,
        evaluation, prediction, and exporting for your model.

    Examples
    --------
    >>> Estimator = xgboost.XGBClassifier()
    >>> BigModelService(estimator=Estimator).start()
    """

    def __init__(self, estimator=None):
        """
        Initial a big model service for JointInference
        :param estimator: Customize estimator
        """

        # 先走 JobBase 初始化，准备好通用配置和模型路径。
        super(BigModelService, self).__init__(estimator=estimator)
        # 推理服务监听地址，默认使用当前主机 IP。
        self.local_ip = self.get_parameters("BIG_MODEL_BIND_IP", get_host_ip())
        # 推理服务监听端口，默认 5000。
        self.port = int(self.get_parameters("BIG_MODEL_BIND_PORT", "5000"))

    def start(self):
        """
        Start inference rest server
        """

        # 允许 estimator 以工厂函数形式传入，这里在真正启动前实例化。
        if callable(self.estimator):
            self.estimator = self.estimator()
        # BigModelService 只接受本地模型目录，不存在则直接报错。
        if not os.path.exists(self.model_path):
            raise FileExistsError(f"{self.model_path} miss")
        else:
            # 启动服务前先加载模型。
            self.estimator.load(self.model_path)
        # 构建并启动 HTTP 推理服务。
        app_server = InferenceServer(model=self, servername=self.job_name,
                                     host=self.local_ip, http_port=self.port)
        app_server.start()

    def train(self, train_data,
              valid_data=None,
              post_process=None,
              **kwargs):
        """todo: no support yet"""

    def inference(self, data=None, post_process=None, **kwargs):
        """
        Inference task for JointInference

        Parameters
        ----------
        data: BaseDataSource
            datasource use for inference, see
            `sedna.datasources.BaseDataSource` for more detail.
        post_process: function or a registered method
            effected after `estimator` inference.
        kwargs: Dict
            parameters for `estimator` inference,
            Like:  `ntree_limit` in Xgboost.XGBClassifier

        Returns
        -------
        inference result
        """

        # 统一把 post_process 解析成真正可调用的 callback。
        callback_func = None
        if callable(post_process):
            callback_func = post_process
        elif post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)

        # 直接调用 estimator.predict 做推理。
        res = self.estimator.predict(data, **kwargs)
        # 若有回调，再对结果做一层后处理。
        if callback_func:
            res = callback_func(res)
        return res


class JointInference(JobBase):
    """
    Sedna provide a framework make sure under the condition of limited
    resources on the edge, difficult inference tasks are offloaded to the
    cloud to improve the overall performance, keeping the throughput.

    Parameters
    ----------
    estimator : Instance
        An instance with the high-level API that greatly simplifies
        machine learning programming. Estimators encapsulate training,
        evaluation, prediction, and exporting for your model.
    hard_example_mining : Dict
        HEM algorithms with parameters which has registered to ClassFactory,
        see `sedna.algorithms.hard_example_mining` for more detail.

    Examples
    --------
    >>> Estimator = keras.models.Sequential()
    >>> ji_service = JointInference(
            estimator=Estimator,
            hard_example_mining={
                "method": "IBT",
                "param": {
                    "threshold_img": 0.9
                }
            }
        )

    Notes
    -----
    Sedna provide an interface call `get_hem_algorithm_from_config` to build
    the `hard_example_mining` parameter from CRD definition.
    """

    def __init__(
            self,
            estimator=None,
            cloud=None,
            drafter=None,
            verifier=None,
            hard_example_mining: dict = None,
            LCReporter_enable: bool = True
        ):
        # 先完成 JobBase 初始化，准备 job_name / worker_name / config / model_path 等通用字段。
        super(JointInference, self).__init__(estimator=estimator)

        # 标记自己在控制面上属于 Joint Inference Service。
        self.job_kind = K8sResourceKind.JOINT_INFERENCE_SERVICE.value
        # 当前机器 IP，用于本地服务或默认 client 配置。
        self.local_ip = get_host_ip()

        # LCReporter 上报消息骨架。
        report_msg = {
            "name": self.worker_name,
            "namespace": self.config.namespace,
            "ownerName": self.job_name,
            "ownerKind": self.job_kind,
            "kind": "inference",
            "results": []
        }
        # 上报周期，默认 30 秒。
        period_interval = int(self.get_parameters("LC_PERIOD", "30"))

        # 允许 benchmark / 本地实验关闭上报线程。
        self.LCReporter_enable = LCReporter_enable

        if self.LCReporter_enable:
            # 启动后台 LCReporter，用于控制面统计。
            self.lc_reporter = LCReporter(
                lc_server=self.config.lc_server,
                message=report_msg,
                period_interval=period_interval
            )
            self.lc_reporter.setDaemon(True)
            self.lc_reporter.start()

        # 允许 estimator 以工厂函数形式传入。
        if callable(self.estimator):
            self.estimator = self.estimator()

        # mining-free / SD 路径只保留正式角色名：drafter / verifier。
        self.drafter = drafter
        self.verifier = verifier
        # 只要两侧角色都存在，就说明当前实例应走 mining-free 工作流。
        self.speculative_runtime = self.drafter is not None and self.verifier is not None
        # 同样支持 drafter / verifier 以工厂函数形式注入。
        if callable(self.drafter):
            self.drafter = self.drafter()
        if callable(self.verifier):
            self.verifier = self.verifier()

        # 一个小工具：判断 `model_path` 是否是 HuggingFace 仓库名。
        check_huggingface_repo = lambda x: bool(re.match(HUGGINGFACE_PATH_PATTERN, x))

        if self.speculative_runtime:
            # mining-free 路径下，范式不直接管理模型，只负责触发 drafter / verifier 各自加载。
            if self.drafter is not None and hasattr(self.drafter, "load"):
                self.drafter.load()
            if self.verifier is not None and hasattr(self.verifier, "load"):
                self.verifier.load()
        else:
            # 传统 joint inference 路径仍然沿用 estimator + cloud 的语义。
            if not os.path.exists(self.model_path) and not check_huggingface_repo(self.model_path):
                raise FileExistsError(f"{self.model_path} miss")
            if self.estimator is not None:
                self.estimator.load(model_url=self.model_path)

        # 若用户没有显式传入 cloud，则在传统路径下自动创建远端大模型客户端。
        # mining-free 路径不走这里，因为 verifier 已经承担了“云侧”角色。
        if cloud is None and not self.speculative_runtime:
            self.remote_ip = self.get_parameters(
            "BIG_MODEL_IP", self.local_ip)
            self.port = int(self.get_parameters("BIG_MODEL_PORT", "5000"))

            self.cloud = ModelClient(
                service_name=self.job_name,
                host=self.remote_ip,
                port=self.port
            )
        else:
            # 若用户显式传了 cloud，则直接复用。
            self.cloud = cloud

        # 传统 joint inference 路径会用到的 HEM 算法实例。
        self.hard_example_mining_algorithm = None
        if not hard_example_mining:
            # 若未显式传入 HEM 配置，则尝试从参数系统中读取。
            hard_example_mining = self.get_hem_algorithm_from_config()
        if hard_example_mining:
            hem = hard_example_mining.get("method", "IBT")
            hem_parameters = hard_example_mining.get("param", {})
            # 通过注册表实例化 HEM 算法。
            self.hard_example_mining_algorithm = ClassFactory.get_cls(
                ClassType.HEM, hem
            )(**hem_parameters)

    @classmethod
    def get_hem_algorithm_from_config(cls, **param):
        """
        get the `algorithm` name and `param` of hard_example_mining from crd

        Parameters
        ----------
        param : Dict
            update value in parameters of hard_example_mining

        Returns
        -------
        dict
            e.g.: {"method": "IBT", "param": {"threshold_img": 0.5}}

        Examples
        --------
        >>> JointInference.get_hem_algorithm_from_config(
                threshold_img=0.9
            )
        {"method": "IBT", "param": {"threshold_img": 0.9}}
        """
        # 从参数系统统一提取 HEM 配置。
        return cls.parameters.get_algorithm_from_api(
            algorithm="HEM",
            **param
        )


    def _get_edge_result(self, data, callback_func, **kwargs):
        # 传统路径：先在边端 estimator 上执行推理。
        edge_result = self.estimator.predict(data, **kwargs)
        # 深拷贝一份给 callback 用，避免 callback 污染原始结果对象。
        res = deepcopy(edge_result)

        # 若用户配置了后处理函数，则在副本上执行。
        if callback_func:
            res = callback_func(res)

        # 上报逻辑独立于业务结果；关闭 reporter 时直接跳过。
        if self.LCReporter_enable:
            # 上报一次边端推理事件。
            self.lc_reporter.update_for_edge_inference()

        # 同时返回“最终结果副本”和“原始边端结果”，保持旧接口兼容。
        return res, edge_result

    def _get_cloud_result(self, data, post_process, **kwargs):
        # 传统路径：通过云端客户端执行推理。
        cloud_result = self.cloud.inference(data, post_process=post_process, **kwargs)

        # 深拷贝一份，避免后续修改原始 cloud_result。
        res = deepcopy(cloud_result)

        if self.LCReporter_enable:
            # 上报一次协同 / 上云推理事件。
            self.lc_reporter.update_for_collaboration_inference()

        return res, cloud_result

    def _check_hem_algorithm(self):
        # 某些模式下必须先有 HEM 算法，否则工作流不完整。
        if self.hard_example_mining_algorithm is None:
            raise ValueError("Hard example mining algorithm is not set.")

    def _supports_token_level_speculative_decoding(self):
        # 范式层只检查 mining-free 所需的最小接口集合。
        # 这里不再要求范式理解算法 payload / session 细节。
        drafter_required = [
            "inference",
            "start_session",
            "step",
            "close_session",
        ]
        # verifier 的职责更轻：单模型推理、session 生命周期、proposal 校验。
        verifier_required = ["inference", "start_session", "verify", "close_session"]
        # 只要接口齐全，就认为 example 层已经接好了正式协议。
        return all(hasattr(self.drafter, name) for name in drafter_required) and all(
            hasattr(self.verifier, name) for name in verifier_required
        )

    def _run_token_level_speculative_decoding(self, data, callback_func=None, **kwargs):
        # 这个方法是当前范式层对 mining-free / speculative decoding 的工作流骨架。
        # 它只负责：
        # 1) 解析模式
        # 2) 打开 session
        # 3) draft -> verify 循环
        # 4) 根据 verifier 返回的 stop/progress/result 决定是否结束
        # inference_mode 优先使用调用方显式传参，其次回退到 drafter 默认配置。
        inference_mode = kwargs.get("inference_mode", getattr(self.drafter, "inference_mode", "collaboration"))

        if inference_mode == "cloud-only":
            # cloud-only：直接走 verifier 的单模型推理路径。
            result = self.verifier.inference(data=data, **kwargs)
            # cloud-only 本质上属于“协同/云端”统计口径。
            if self.LCReporter_enable:
                self.lc_reporter.update_for_collaboration_inference()
            # callback 依然统一应用在最终返回值上。
            if callback_func:
                result = callback_func(result)
            # 保持历史 JointInference 四元组返回结构。
            return [False, result, None, result]

        if inference_mode == "edge-only":
            # edge-only：直接走 drafter 的单模型推理路径。
            edge_result = self.drafter.inference(
                data=data,
                **kwargs,
            )
            # edge-only 计入边端推理统计。
            if self.LCReporter_enable:
                self.lc_reporter.update_for_edge_inference()
            # callback 在最终返回前统一处理。
            result = callback_func(edge_result) if callback_func else edge_result
            # 保持历史 JointInference 四元组返回结构。
            return [False, result, edge_result, None]

        # collaboration：先分别创建 drafter / verifier session。
        draft_session = self.drafter.start_session(data=data, **kwargs)
        verify_session = self.verifier.start_session(data=data, draft_session=draft_session, **kwargs)
        # request 主要供 finally 中 close_session 使用。
        request = draft_session.get("request", data if isinstance(data, dict) else {"query": data})

        try:
            # 这里只保留一个 round_index，作为范式层自己的循环计数。
            round_index = 0
            collaboration_result = None
            feedback = None
            while True:
                # 每进入一轮就自增一次，便于后续排查流程卡在哪一轮。
                round_index += 1
                # 第一步：drafter 生成本轮 proposal。
                draft_payload = self.drafter.step(
                    draft_session,
                    feedback=feedback,
                    **kwargs
                )

                # 第二步：verifier 对 proposal 做校验/纠正。
                verify_payload = self.verifier.verify(
                    verify_session,
                    draft_output=draft_payload,
                    **kwargs
                )
                control = {
                    "stop": bool((verify_payload or {}).get("stop", False)),
                    "progress": bool((verify_payload or {}).get("progress", True)),
                }
                if (verify_payload or {}).get("result") is not None:
                    collaboration_result = verify_payload.get("result")
                feedback = (verify_payload or {}).get("feedback")
                draft_session["_pending_feedback"] = feedback

                # 若算法层告诉范式“本轮没有推进”，则直接中止，避免死循环。
                if not bool(control.get("progress", True)):
                    raise RuntimeError(
                        "Speculative decoding made no progress according to algorithm runtime state."
                    )

                # 范式层只根据 verifier 暴露的 stop 信号决定是否退出循环。
                if bool(control.get("stop", False)):
                    break
        finally:
            # 无论流程中间是否异常，session 都必须被回收。
            # 这里使用 finally 是为了防止异常导致 session 泄漏。
            draft_close_result = self.drafter.close_session(draft_session, request=request)
            self.verifier.close_session(verify_session, request=request)
            if collaboration_result is None and draft_close_result is not None:
                collaboration_result = draft_close_result

        if collaboration_result is None:
            raise RuntimeError(
                "Speculative decoding stopped without returning a final result."
            )

        # collaboration 完成后按协同推理统计一次。
        if self.LCReporter_enable:
            self.lc_reporter.update_for_collaboration_inference()
        # 若配置了 callback，则统一在最终结果上应用。
        result = callback_func(collaboration_result) if callback_func else collaboration_result
        # collaboration 在旧接口里不单独返回 edge/cloud 原始结果。
        return [False, result, None, None]

    def inference(self, data=None, post_process=None, **kwargs):
        """
        Inference task with JointInference

        Parameters
        ----------
        data: BaseDataSource
            datasource use for inference, see
            `sedna.datasources.BaseDataSource` for more detail.
        post_process: function or a registered method
            effected after `estimator` inference.
        kwargs: Dict
            parameters for `estimator` inference,
            Like:  `ntree_limit` in Xgboost.XGBClassifier

        Returns
        -------
        if is hard sample : bool
        inference result : object
        result from little-model : object
        result from big-model: object
        """

        # 统一把 post_process 解析成真正可调用的 callback。
        callback_func = None
        if callable(post_process):
            callback_func = post_process
        elif post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)

        # mining_mode 是 JointInference 的总分支开关：
        # - inference-then-mining
        # - mining-then-inference
        # - self-design
        # - mining-free
        mining_mode = kwargs.get("mining_mode", "inference-then-mining")

        # 传统 joint inference 路径统一维护这几个返回槽位。
        is_hard_example = False
        # 这个历史变量目前并未启用，只保留在旧分支语义里。
        sepeculative_decoding = False

        # 预先初始化边端/云端结果槽位，便于末尾统一返回。
        edge_result, cloud_result = None, None

        if mining_mode == "inference-then-mining":
            # 先在边端推理，再让 HEM 判断是否需要上云。
            res, edge_result = self._get_edge_result(data, callback_func, **kwargs)

            self._check_hem_algorithm()

            # HEM 输入的是边端结果副本，由算法自行判断是否属于困难样本。
            is_hard_example = self.hard_example_mining_algorithm(res)
            if is_hard_example:
                # 命中困难样本后，再向云端请求更强模型结果。
                res, cloud_result = self._get_cloud_result(data, post_process=post_process, **kwargs)

        elif mining_mode == "mining-then-inference":
            # 先做 HEM，再根据结果决定去 edge 还是 cloud。
            self._check_hem_algorithm()

            # 该模式下 HEM 直接基于原始输入数据做预判。
            is_hard_example = self.hard_example_mining_algorithm(data)
            if is_hard_example:
                if not sepeculative_decoding:
                    # 当前仍沿用传统 cloud 路径；历史 speculative 分支未启用。
                    res, cloud_result = self._get_cloud_result(data, post_process=post_process, **kwargs)
                else:
                    # 这个历史分支目前未启用。
                    pass
            else:
                # 非困难样本直接在边端完成推理。
                res, edge_result = self._get_edge_result(data, callback_func, **kwargs)

        elif mining_mode == "self-design":
            # self-design：完全跳过范式内置路由逻辑，直接调用 estimator。
            res = self.estimator.predict(data, **kwargs)
            # self-design 返回值也允许走统一 callback。
            return callback_func(res) if callback_func else res

        elif mining_mode == "mining-free":
            # mining-free：不再走 HEM，不再做“困难样本路由”，
            # 而是直接进入 drafter/verifier 的多轮工作流。
            if self._supports_token_level_speculative_decoding():
                return self._run_token_level_speculative_decoding(
                    data=data,
                    callback_func=callback_func,
                    **kwargs,
                )
            raise ValueError(
                "Mining-free mode requires drafter/verifier runtimes "
                "implementing inference/start_session/step/verify/close_session."
            )

        else:
            # 非法 mining_mode 明确报错，并列出允许值。
            raise ValueError(
                "Mining Mode must be in "
                "['mining-then-inference', 'inference-then-mining', "
                "'self-design', 'mining-free']"
            )

        # 传统 joint inference 路径统一返回四元组：
        # [是否困难样本, 最终结果, 边端结果, 云端结果]
        return [is_hard_example, res, edge_result, cloud_result]
