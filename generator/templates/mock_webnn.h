//* Copyright 2017 The Dawn Authors
//*
//* Licensed under the Apache License, Version 2.0 (the "License");
//* you may not use this file except in compliance with the License.
//* You may obtain a copy of the License at
//*
//*     http://www.apache.org/licenses/LICENSE-2.0
//*
//* Unless required by applicable law or agreed to in writing, software
//* distributed under the License is distributed on an "AS IS" BASIS,
//* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//* See the License for the specific language governing permissions and
//* limitations under the License.

#ifndef MOCK_WEBNN_H
#define MOCK_WEBNN_H

#include <webnn/webnn_proc_table.h>
#include <webnn/webnn.h>
#include <gmock/gmock.h>

#include <memory>

// An abstract base class representing a proc table so that API calls can be mocked. Most API calls
// are directly represented by a delete virtual method but others need minimal state tracking to be
// useful as mocks.
class ProcTableAsClass {
    public:
        virtual ~ProcTableAsClass();

        void GetProcTableAndDevice(WebnnProcTable* table);

        // Creates an object that can be returned by a mocked call as in WillOnce(Return(foo)).
        // It returns an object of the write type that isn't equal to any previously returned object.
        // Otherwise some mock expectation could be triggered by two different objects having the same
        // value.
        {% for type in by_category["object"] %}
            {{as_cType(type.name)}} GetNew{{type.name.CamelCase()}}();
        {% endfor %}

        {% for type in by_category["object"] %}
            {% for method in type.methods if len(method.arguments) < 10 and not has_callback_arguments(method) %}
                virtual {{as_cType(method.return_type.name)}} {{as_MethodSuffix(type.name, method.name)}}(
                    {{-as_cType(type.name)}} {{as_varName(type.name)}}
                    {%- for arg in method.arguments -%}
                        , {{as_annotated_cType(arg)}}
                    {%- endfor -%}
                ) = 0;
            {% endfor %}
            virtual void {{as_MethodSuffix(type.name, Name("reference"))}}({{as_cType(type.name)}} self) = 0;
            virtual void {{as_MethodSuffix(type.name, Name("release"))}}({{as_cType(type.name)}} self) = 0;
        {% endfor %}

	// Special cased mockable methods
        virtual void OnCompilationComputeCallback(WebnnCompilation self,
                                WebnnNamedInputs inputs,
                                WebnnComputeCallback callback,
                                void* userdata, WebnnNamedOutputs outputs) = 0; 	

	virtual void  OnModelCompileCallback(WebnnModel self, WebnnCompileCallback callback,
                          void* userdata,
                          WebnnCompilationOptions const * options) = 0;
 
	virtual bool OnNeuralNetworkContextPopErrorScopeCallback(WebnnNeuralNetworkContext 
		         neuralNetworkContext,
                         WebnnErrorCallback callback, void * userdata) = 0;

	void CompilationCompute(WebnnCompilation self, 
			        WebnnNamedInputs inputs, 
				WebnnComputeCallback callback, 
				void* userdata, WebnnNamedOutputs outputs);

	void ModelCompile(WebnnModel self, WebnnCompileCallback callback, 
			  void* userdata, 
			  WebnnCompilationOptions const * options);

	bool NeuralNetworkContextPopErrorScope(WebnnNeuralNetworkContext neuralNetworkContext, 
			                       WebnnErrorCallback callback, void * userdata);

	void NeuralNetworkContextSetUncapturedErrorCallback(WebnnNeuralNetworkContext neuralNetworkContext, 
			                       WebnnErrorCallback callback, void * userdata);

	struct Object {
            ProcTableAsClass* procs = nullptr;
	    WebnnComputeCallback computeCallback = nullptr;
	    WebnnCompileCallback compileCallback = nullptr;
	    WebnnErrorCallback errorCallback = nullptr;
            void* userdata = 0;
        };

    private:
        // Remembers the values returned by GetNew* so they can be freed.
        std::vector<std::unique_ptr<Object>> mObjects;
};

class MockProcTable : public ProcTableAsClass {
    public:
        MockProcTable();
        ~MockProcTable() override;

        void IgnoreAllReleaseCalls();

        {% for type in by_category["object"] %}
            {% for method in type.methods if len(method.arguments) < 10 and not has_callback_arguments(method) %}
                MOCK_METHOD({{as_cType(method.return_type.name)}},{{" "}}
                    {{-as_MethodSuffix(type.name, method.name)}}, (
                        {{-as_cType(type.name)}} {{as_varName(type.name)}}
                        {%- for arg in method.arguments -%}
                            , {{as_annotated_cType(arg)}}
                        {%- endfor -%}
                    ), (override));
            {% endfor %}

            MOCK_METHOD(void, {{as_MethodSuffix(type.name, Name("reference"))}}, ({{as_cType(type.name)}} self), (override));
            MOCK_METHOD(void, {{as_MethodSuffix(type.name, Name("release"))}}, ({{as_cType(type.name)}} self), (override));
        {% endfor %}

	 MOCK_METHOD(void, 
		     OnCompilationComputeCallback, 
		     (WebnnCompilation self,
                     WebnnNamedInputs inputs,
                     WebnnComputeCallback callback,
                     void* userdata, WebnnNamedOutputs outputs), (override));

	 MOCK_METHOD(void,
                     OnModelCompileCallback,
                     (WebnnModel self, 
		     WebnnCompileCallback callback,
                     void* userdata,
                     WebnnCompilationOptions const * options), (override));

	 MOCK_METHOD(bool,
                     OnNeuralNetworkContextPopErrorScopeCallback,
                     (WebnnNeuralNetworkContext neuralNetworkContext,
                      WebnnErrorCallback callback, void * userdata), (override));


};

#endif  // MOCK_WEBNN_H
